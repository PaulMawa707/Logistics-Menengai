# app_wialon.py

import streamlit as st
import pandas as pd
import requests, json, time
from datetime import datetime, timedelta
import pytz, pdfplumber, numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd

REQUIRED_COLUMNS = ["REP","CUSTOMER ID","CUSTOMER NAME","LOCATION","COORDINATES","INVOICE NO.","AMOUNT","TONNAGE"]

def extract_coordinates(coord_str):
    try:
        if "LAT:" in coord_str and "LONG:" in coord_str:
            parts = coord_str.split("LONG:")
            lat = float(parts[0].replace("LAT:","").strip())
            lon = float(parts[1].strip())
            return lat, lon
    except Exception:
        pass
    return None, None

def read_pdf_to_df(pdf_file):
    def clean_column_names(cols):
        return [str(c).replace("\n"," ").strip() for c in cols if c]

    all_rows, header = [], None
    with pdfplumber.open(pdf_file) as pdf:
        for i, pg in enumerate(pdf.pages):
            for tbl in pg.extract_tables() or []:
                if i==0 and header is None:
                    for row in tbl:
                        if row and row[0] and not row[0].startswith("Sales Order Booking"):
                            header = clean_column_names(row)
                            break
                    data = tbl[1:] if header else []
                else:
                    data = tbl
                all_rows += [r for r in data if any(str(c).strip() for c in r)]

    if not header or not all_rows:
        raise ValueError("PDF parsing failed — no valid data found.")

    df = pd.DataFrame(all_rows, columns=clean_column_names(header))
    df = df[~df.apply(lambda r: any(str(c).strip().upper() in REQUIRED_COLUMNS for c in r), axis=1)]
    df = df[~df.astype(str).apply(lambda r: r.str.contains("Fixed|Driver Sign|Mileage|Cartons", case=False).any(), axis=1)]
    df = df[[c for c in df.columns if c.strip().upper() in REQUIRED_COLUMNS]]
    df.columns = [c.strip().upper() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).replace(r"\s*\n\s*", " ", regex=True).str.strip()
    df = df[df["CUSTOMER ID"].notna()].dropna(how="all").reset_index(drop=True)

    df[["LAT","LONG"]] = df["COORDINATES"].apply(lambda x: pd.Series(extract_coordinates(x)))
    df = df.dropna(subset=["LAT","LONG"]).reset_index(drop=True)

    coords = np.radians(df[["LAT","LONG"]])
    df["Cluster"] = DBSCAN(eps=5/6371.0088, min_samples=1, algorithm="ball_tree", metric="haversine").fit(coords).labels_

    counties = gpd.read_file("kenya-counties-simplified.geojson").to_crs("EPSG:4326")
    gdf_pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["LONG"], df["LAT"])], crs="EPSG:4326")
    gdf = gpd.sjoin(gdf_pts, counties[["shapeName","geometry"]], how="left", predicate="within")
    gdf = gdf.rename(columns={"shapeName":"Correct County"}).sort_values("Correct County").reset_index(drop=True)

    st.subheader("🗺️ Locations Matched with Counties")
    st.dataframe(gdf[["REP","CUSTOMER ID","CUSTOMER NAME","LOCATION","COORDINATES","INVOICE NO.","AMOUNT","TONNAGE","LAT","LONG","Correct County"]])
    return gdf

def assign_assets_ui(gdf):
    counties = sorted(gdf["Correct County"].dropna().unique())
    st.subheader("🚚 Assign Assets to Counties")
    county_asset = {}
    for c in counties:
        county_asset[c] = st.text_input(f"Asset for {c}", key=f"asset_{c}")
    return county_asset

def optimize_route(orders, sid):
    """Optimize route using Wialon's route optimization API"""
    url = f"https://hst-api.wialon.com/wialon/ajax.html?sid={sid}"
    
    # Group orders by resource (vehicle)
    orders_by_resource = {}
    for order in orders:
        resource = order['rp']
        if resource not in orders_by_resource:
            orders_by_resource[resource] = []
        orders_by_resource[resource].append(order)
    
    optimized_orders = []
    for resource, resource_orders in orders_by_resource.items():
        if len(resource_orders) < 2:
            optimized_orders.extend(resource_orders)
            continue
            
        # Create route optimization request
        optimization_params = {
            "orders": [order['id'] for order in resource_orders],
            "resource": resource,
            "flags": 1,  # Use actual roads for routing
            "departure": resource_orders[0]['tf'],
            "path_type": "optimal",  # Find optimal path between points
            "optimization_type": "time"  # Optimize for minimum time
        }
        
        try:
            # Call route optimization API
            resp = requests.post(url, params={
                "svc": "order/optimize",
                "params": json.dumps(optimization_params)
            })
            result = resp.json()
            
            if isinstance(result, dict) and 'orders' in result:
                # Update order sequence based on optimization
                for idx, order_id in enumerate(result['orders']):
                    for order in resource_orders:
                        if order['id'] == order_id:
                            order['sequence'] = idx + 1
                            break
                optimized_orders.extend(resource_orders)
            else:
                st.warning(f"Route optimization failed for resource {resource}. Using original order.")
                optimized_orders.extend(resource_orders)
        except Exception as e:
            st.warning(f"Route optimization error for resource {resource}: {str(e)}")
            optimized_orders.extend(resource_orders)
            
    return optimized_orders

def update_route_paths(orders, sid):
    """Update route paths to follow roads"""
    url = f"https://hst-api.wialon.com/wialon/ajax.html?sid={sid}"
    
    for order in orders:
        try:
            # Request route update to follow roads
            route_params = {
                "id": order['id'],
                "path_type": "optimal",  # Use optimal path that follows roads
                "flags": 1  # Use actual roads
            }
            
            resp = requests.post(url, params={
                "svc": "order/route_update",
                "params": json.dumps(route_params)
            })
            
            result = resp.json()
            if isinstance(result, dict) and 'error' in result:
                st.warning(f"Route update failed for order {order['n']}: {result['error']}")
        except Exception as e:
            st.warning(f"Route update error for order {order['n']}: {str(e)}")
        
        time.sleep(1)  # Rate limiting

def convert_to_orders(gdf, tf, tt, county_asset):
    orders = []
    for _, r in gdf.iterrows():
        lat, lon = extract_coordinates(r["COORDINATES"])
        if lat is None: continue
        w = int(float(r.get("TONNAGE",0)) * 1000) if r.get("TONNAGE") else 0
        asset = county_asset.get(r["Correct County"], "")
        order = {
            "itemId": 25601229,
            "id": 0,
            "n": r["CUSTOMER NAME"],
            "oldOrderId": 0,
            "oldOrderFiles": [],
            "p": {
                "n": r["CUSTOMER NAME"],
                "a": f"{r['Correct County']} (LAT:{lat}, LONG:{lon})",
                "w": w,
                "c": r["AMOUNT"],
                "cid": str(r["CUSTOMER ID"]),
                "uic": str(r["INVOICE NO"]),
                "cm": "Handle with care"
            },
            "rp": asset or "Unassigned",
            "tf": tf,
            "tt": tt,
            "trt": 600,
            "r": 20,
            "y": lat,
            "x": lon,
            "tz": 3,
            "ej": {},
            "callMode": "create",
            "dp": [],
            "cf": {
                "delivery_notes": "",
                "payment_status": ""
            },
            "routing_mode": "optimal",  # Use optimal routing that follows roads
            "path_type": "optimal"  # Ensure route follows roads
        }
        orders.append(order)
    return orders

def login_to_wialon(token):
    res = requests.get("https://hst-api.wialon.com/wialon/ajax.html", params={"svc":"token/login","params":json.dumps({"token":token})})
    data = res.json()
    if "eid" not in data:
        raise Exception(f"Login failed: {data}")
    return data["eid"]

def send_orders(orders, sid):
    """Send orders and optimize routes"""
    url = f"https://hst-api.wialon.com/wialon/ajax.html?sid={sid}"
    created_orders = []
    
    # First create all orders
    for o in orders:
        resp = requests.post(url, params={"svc":"order/update","params":json.dumps(o)})
        result = resp.json()
        
        if isinstance(result, dict) and 'error' in result:
            st.error(f"❌ Failed to create order for {o['n']}: {result['error']}")
        else:
            o['id'] = result  # Store the created order ID
            created_orders.append(o)
            st.success(f"✅ Created order for {o['n']} → ID: {result}")
        
        time.sleep(1)
    
    if created_orders:
        # Optimize routes for created orders
        st.info("Optimizing routes...")
        optimized_orders = optimize_route(created_orders, sid)
        
        # Update routes to follow roads
        st.info("Updating routes to follow roads...")
        update_route_paths(optimized_orders, sid)
        
        st.success("✅ Routes optimized and updated to follow roads")

def run_wialon_uploader():
    st.set_page_config(layout="wide")
    st.title("📦 Wialon Logistics Order Uploader")

    with st.form("upload_form"):
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        date_ = st.date_input("Delivery Date")
        token = st.text_input("Wialon Token", type="password")
        submit = st.form_submit_button("Process Orders")

    if submit:
        if not pdf or not token:
            st.error("Please upload a PDF and enter your token.")
            return

        tz = pytz.timezone("Africa/Nairobi")
        start = tz.localize(datetime.combine(date_, datetime.min.time()))
        tf, tt = int(start.timestamp()), int((start + timedelta(days=1)).timestamp())

        try:
            gdf = read_pdf_to_df(pdf)
            assets = assign_assets_ui(gdf)
            orders = convert_to_orders(gdf, tf, tt, assets)
            st.info(f"{len(orders)} orders prepared. Sending now…")
            sid = login_to_wialon(token)
            send_orders(orders, sid)
        except Exception as e:
            st.error(f"❌ {e}")

if __name__ == "__main__":
    run_wialon_uploader()
