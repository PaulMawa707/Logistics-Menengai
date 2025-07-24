import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import pdfplumber
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
import os
import base64

# Set timezone environment variable
os.environ['TZ'] = 'Africa/Nairobi'
try:
    time.tzset()  # Unix-specific, will fail on Windows
except AttributeError:
    pass  # Skip on Windows

st.set_page_config(page_title="Wialon Logistics Uploader", layout="wide")

# Convert image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background image
def set_background():
    background_image = get_base64_image("pexels-pixabay-236722.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Show logo at top-right corner
def show_logo_top_right(image_path, width=120):
    logo_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div></div> <!-- empty left spacer -->
            <div style="margin-right: 1rem;">
                <img src="data:image/png;base64,{logo_base64}" width="{width}">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ✅ Apply background and logo
set_background()
show_logo_top_right("CT-Logo.jpg", width=120)  # 

st.markdown("<br>", unsafe_allow_html=True)

def extract_coordinates(coord_str):
    try:
        if "LAT:" in coord_str and "LONG:" in coord_str:
            parts = coord_str.split("LONG:")
            latitude = float(parts[0].replace("LAT:", "").strip().replace(" ", ""))
            longitude = float(parts[1].strip().replace(" ", ""))
            return latitude, longitude
    except:
        pass
    return None, None

def read_pdf_to_df(pdf_file):
    def clean_column_names(columns):
        return [str(col).replace('\n', ' ').strip() if col else "" for col in columns]

    all_rows = []
    header = None
    truck_number = None

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if "Truck Number:" in text:
                for line in text.split('\n'):
                    if "Truck Number:" in line:
                        truck_number = line.split("Truck Number:")[1].split("Gate Pass")[0].strip()
                        break
            tables = page.extract_tables()
            if not tables:
                continue
            for table in tables:
                if not table:
                    continue
                if i == 0 and not header:
                    for row in table:
                        if row and row[0] and not row[0].startswith("Sales Order Booking Delivery Sheet"):
                            header = clean_column_names(row)
                            break
                    data = table[1:] if header else []
                else:
                    data = table
                data = [row for row in data if any(cell and str(cell).strip() for cell in row)]
                all_rows.extend(data)

    if not header or not all_rows:
        raise ValueError("Could not extract table properly.")

    df_cleaned = pd.DataFrame(all_rows, columns=clean_column_names(header))
    df_cleaned.columns = [col.strip().upper() for col in df_cleaned.columns]

    for col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()

    df_cleaned = df_cleaned.dropna(how='all')
    df_cleaned = df_cleaned[df_cleaned['CUSTOMER ID'].notna()].reset_index(drop=True)

    # Clean and convert numeric values
    df_cleaned['TONNAGE'] = df_cleaned['TONNAGE'].astype(str).str.replace(',', '').str.strip()
    df_cleaned['TONNAGE'] = pd.to_numeric(df_cleaned['TONNAGE'], errors='coerce').fillna(0)
    
    # Clean and convert Amount - handle comma-formatted numbers
    df_cleaned['AMOUNT'] = df_cleaned['AMOUNT'].astype(str).str.replace(',', '').str.strip()
    df_cleaned['AMOUNT'] = pd.to_numeric(df_cleaned['AMOUNT'], errors='coerce').fillna(0)

    # Group by Customer ID and sum weights and costs
    df_grouped = df_cleaned.groupby(['CUSTOMER ID', 'CUSTOMER NAME', 'LOCATION', 'LOCATION COORDINATES', 'REP'], as_index=False).agg({
        'TONNAGE': 'sum',
        'AMOUNT': 'sum',
        'INVOICE NO.': lambda x: ', '.join(str(i) for i in x if pd.notna(i))
    })

    df_grouped[['LAT', 'LONG']] = df_grouped['LOCATION COORDINATES'].apply(lambda x: pd.Series(extract_coordinates(x)))
    df_grouped = df_grouped.dropna(subset=['LAT', 'LONG'])

    coords_rad = np.radians(df_grouped[['LAT', 'LONG']])
    epsilon = 5 / 6371.0088
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    df_grouped['Cluster'] = db.labels_

    df_grouped = df_grouped.sort_values(by=['Cluster', 'LAT', 'LONG']).reset_index(drop=True)

    geojson_path = "kenya-counties-simplified.geojson"
    if not os.path.exists(geojson_path):
        st.error(f"GeoJSON file '{geojson_path}' not found.")
        return None, None

    gdf_points = gpd.GeoDataFrame(df_grouped, geometry=[Point(xy) for xy in zip(df_grouped['LONG'], df_grouped['LAT'])], crs="EPSG:4326")
    kenya_counties = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    gdf_joined = gpd.sjoin(gdf_points, kenya_counties[['shapeName', 'geometry']], how="left", predicate="within")
    gdf_joined = gdf_joined.rename(columns={'shapeName': 'Correct County'})
    gdf_joined = gdf_joined.sort_values(by='Correct County').reset_index(drop=True)

    return gdf_joined, truck_number

def read_asset_id_from_excel(excel_file, truck_number):
    df = pd.read_excel(excel_file)
    df.columns = [col.strip().lower() for col in df.columns]

    if "reportname" not in df.columns or "itemid" not in df.columns:
        raise ValueError("Excel file must contain 'ReportName' and 'itemId' columns.")

    match = df[df["reportname"].str.upper().str.contains(truck_number.upper(), na=False)]

    if not match.empty:
        return int(match.iloc[0]["itemid"]), match.iloc[0]["reportname"]

    return None, None

def send_orders_and_create_route(token, resource_id, unit_id, vehicle_name, df_grouped, tf, tt):
    try:
        # Base URL for Wialon API
        base_url = "https://hst-api.wialon.com/wialon/ajax.html"

        # First login with token to get session
        login_payload = {
            "svc": "token/login",
            "params": json.dumps({
                "token": str(token).strip()
            })
        }
        
        st.info("Logging in with token...")
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        login_response = requests.post(base_url, data=login_payload, headers=headers)
        login_result = login_response.json()
        session_id = login_result['eid']

        # Define MORL warehouse location
        morl_lat = -0.28802969095623043
        morl_lon = 36.04494759379902

        # Calculate distances from MORL to each delivery point
        def calculate_distance(lat1, lon1, lat2, lon2):
            from math import sin, cos, sqrt, atan2, radians
            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c

            return distance

        # Add distance from MORL to each point
        df_grouped['Distance_From_MORL'] = df_grouped.apply(
            lambda row: calculate_distance(morl_lat, morl_lon, row['LAT'], row['LONG']),
            axis=1
        )

        # Sort by distance from MORL
        df_grouped = df_grouped.sort_values('Distance_From_MORL').reset_index(drop=True)

        # Step 1: Create orders for optimization
        orders = []
        
        # Add MORL as starting point with ID 0
        morl_coords = f"{morl_lat}, {morl_lon}"
        orders.append({
            "y": morl_lat,
            "x": morl_lon,
            "tf": tf,
            "tt": tt,
            "n": "MORL",
            "f": 1,
            "r": 2500,
            "id": 0,
            "p": {
                "ut": 0,
                "rep": True,
                "w": 0,
                "v": 0,
                "pr": 0,
                "criterions": {
                    "max_late": 0,
                    "use_unloading_late": 0
                },
                "a": f"MORL ({morl_coords})"  # Add coordinates to address
            },
            "cmp": {"unitRequirements": {"values": []}}
        })

        # Add delivery points with sequential IDs
        priority_dict = {str(unit_id): {"0": 0}}

        for idx, row in df_grouped.iterrows():
            try:
                weight_kg = int(float(row.get('TONNAGE', 0)) * 1000)
            except:
                weight_kg = 0

            # Format coordinates for address field
            coords = f"{row['LAT']}, {row['LONG']}"
            location = f"{row['LOCATION']} ({coords})"

            order_id = idx + 1
            orders.append({
                "y": float(row['LAT']),
                "x": float(row['LONG']),
                "tf": tf,
                "tt": tt,
                "n": row['CUSTOMER NAME'],
                "f": 0,
                "r": 2500,
                "id": order_id,
                "p": {
                    "ut": 900,
                    "rep": True,
                    "w": weight_kg,
                    "v": 0,
                    "pr": idx + 1,
                    "criterions": {
                        "max_late": 0,
                        "use_unloading_late": 0
                    },
                    "a": location  # Add coordinates to address
                },
                "cmp": {"unitRequirements": {"values": []}}
            })
            priority_dict[str(unit_id)][str(order_id)] = idx + 1

        # Add MORL as end point with last ID
        final_id = len(orders) + 1
        orders.append({
            "y": morl_lat,
            "x": morl_lon,
            "tf": tf,
            "tt": tt,
            "n": "MORL",
            "f": 2,
            "r": 2500,
            "id": final_id,
            "p": {
                "ut": 0,
                "rep": True,
                "w": 0,
                "v": 0,
                "pr": len(df_grouped) + 2,
                "criterions": {
                    "max_late": 0,
                    "use_unloading_late": 0
                },
                "a": f"MORL ({morl_coords})"  # Add coordinates to address
            },
            "cmp": {"unitRequirements": {"values": []}}
        })
        priority_dict[str(unit_id)][str(final_id)] = len(df_grouped) + 2

        # Step 2: Optimize route
        optimize_payload = {
            "svc": "order/optimize",
            "params": json.dumps({
                "itemId": int(resource_id),
                "orders": orders,
                "warehouses": [
                    {
                        "id": 0,
                        "y": morl_lat,
                        "x": morl_lon,
                        "n": "MORL",
                        "f": 1,
                        "a": f"MORL ({morl_coords})"  # Add coordinates to address
                    }
                ],
                "flags": 0x1 | 0x2 | 0x20 | 0x40,
                "units": [int(unit_id)],
                "gis": {
                    "addPoints": 1,
                    "provider": 2,
                    "speed": 40,
                    "cityJams": 1,
                    "countryJams": 1,
                    "avoid": ["tolls"],
                    "traffic_model": "best_guess",
                    "departure_time": tf,
                    "mode": "driving",
                    "calculate_point_to_point": 1,
                    "use_route_points": 1
                },
                "priority": priority_dict,
                "criterions": {
                    "penalties_profile": "cost_effective",
                    "max_order_count": len(orders),
                    "split_intervals": 1,
                    "respect_priority": 1,
                    "optimize_sequence": 0
                },
                "pf": {
                    "n": "MORL",
                    "y": morl_lat,
                    "x": morl_lon,
                    "a": f"MORL ({morl_coords})"  # Add coordinates to address
                },
                "pt": {
                    "n": "MORL",
                    "y": morl_lat,
                    "x": morl_lon,
                    "a": f"MORL ({morl_coords})"  # Add coordinates to address
                },
                "tf": tf,
                "tt": tt
            }),
            "sid": session_id
        }

        st.info("Optimizing route...")
        optimize_response = requests.post(base_url, data=optimize_payload)
        optimize_result = optimize_response.json()

        # Check for optimization errors
        if 'error' in optimize_result:
            if optimize_result['error'] != 0:
                error_msg = optimize_result.get('reason', 'Unknown error during optimization')
                st.error(f"Optimization failed: {error_msg}")
                if 'details' in optimize_result:
                    st.write("Error details:", optimize_result['details'])
                return {"error": optimize_result['error'], "message": error_msg}

        # Extract optimized orders and route summary
        optimized_orders = []
        route_summary = None

        # The response structure shows orders directly in the unit_id object
        if isinstance(optimize_result, dict):
            unit_key = str(unit_id)
            if unit_key in optimize_result and 'orders' in optimize_result[unit_key]:
                optimized_orders = optimize_result[unit_key]['orders']
                if 'routes' in optimize_result[unit_key]:
                    route_summary = optimize_result[unit_key]['routes'][0]

        if not optimized_orders:
            st.error("No optimized orders found in response")
            return {"error": 1, "message": "No optimized orders found"}

        # Create a mapping of coordinates from the original orders
        coord_map = {}
        for idx, row in df_grouped.iterrows():
            coord_map[row['CUSTOMER NAME']] = {
                'y': float(row['LAT']),
                'x': float(row['LONG'])
            }

        # Add warehouse coordinates
        coord_map['MORL)'] = {'y': morl_lat, 'x': morl_lon}
        

        # Step 3: Create final route with routing information
        route_orders = []
        current_time = int(time.time())
        route_id = current_time

        # Create a list to store order names in sequence
        order_names = ['MORL']
        for idx, row in df_grouped.iterrows():
            order_names.append(row['CUSTOMER NAME'])
        order_names.append('MORL')

        # Process optimized orders
        for idx, order in enumerate(optimized_orders):
            order_name = order_names[idx] if idx < len(order_names) else f"Stop {idx}"
            coords = coord_map.get(order_name, {'y': morl_lat, 'x': morl_lon})
            
            is_start = idx == 0
            is_end = idx == len(optimized_orders) - 1
            order_flag = 1 if is_start else (2 if is_end else 0)
            
            # Get weight and cost from original orders
            weight_kg = 0
            cost = 0
            if not (is_start or is_end):
                try:
                    # Find the customer data in df_grouped
                    customer_data = df_grouped[df_grouped['CUSTOMER NAME'] == order_name]
                    if not customer_data.empty:
                        # Convert tonnage to kilograms
                        weight_kg = int(float(customer_data.iloc[0]['TONNAGE']) * 1000)
                        # Get amount directly
                        cost = float(customer_data.iloc[0]['AMOUNT'])
                except Exception as e:
                    weight_kg = 0
                    cost = 0

            # Format coordinates for address
            location = f"{order_name} ({coords['y']}, {coords['x']})"
            if not (is_start or is_end):
                try:
                    location_data = df_grouped[df_grouped['CUSTOMER NAME'] == order_name]
                    if not location_data.empty:
                        location = f"{location_data.iloc[0]['LOCATION']} ({coords['y']}, {coords['x']})"
                except:
                    pass
            
            if isinstance(order, dict):
                order_id = order.get('id', idx)
                order_tm = order.get('tm', current_time)
                order_ml = order.get('ml', 0)
                order_p = order.get('p', '')
            else:
                continue

            # Calculate distance from previous point
            if idx > 0:
                prev_coords = coord_map.get(order_names[idx-1], {'y': morl_lat, 'x': morl_lon})
                distance = calculate_distance(
                    prev_coords['y'], prev_coords['x'],
                    coords['y'], coords['x']
                )
                mileage = int(distance * 1000)  # Convert km to meters
            else:
                mileage = 0
            
            route_order = {
                "uid": int(unit_id),
                "id": order_id,
                "n": order_name,
                "p": {
                    "ut": 0 if (is_start or is_end) else 900,
                    "rep": True,
                    "w": str(weight_kg),
                    "c": str(int(cost)),
                    "r": {
                        "vt": order_tm,
                        "ndt": 60,
                        "id": route_id,
                        "i": order_id,
                        "m": mileage,
                        "t": 0
                    },
                    "u": int(unit_id),
                    "a": location,
                    "weight": str(weight_kg),
                    "cost": str(int(cost))
                },
                "f": order_flag,
                "tf": tf,
                "tt": tt,
                "r": 2500,
                "y": coords['y'],
                "x": coords['x'],
                "s": 0,
                "sf": 0,
                "trt": 0,
                "st": current_time,
                "cnm": 0,
                "rp": order_p,
                "ej": {},
                "cf": {},
                "cmp": {"unitRequirements": {"values": []}},
                "gfn": {"geofences": {}},
                "callMode": "create",
                "u": int(unit_id),
                "weight": str(weight_kg),
                "cost": str(int(cost)),
                "cargo": {
                    "weight": str(weight_kg),
                    "cost": str(int(cost))
                }
            }
            route_orders.append(route_order)

        # Calculate total mileage and cost
        total_mileage = sum(order['p']['r']['m'] for order in route_orders)
        total_cost = sum(float(order['p']['c']) for order in route_orders if order['f'] == 0)
        total_weight = sum(int(order['p']['w']) for order in route_orders if order['f'] == 0)

        # Create the final route
        batch_payload = {
            "svc": "core/batch",
            "params": json.dumps({
                "params": [{
                    "svc": "order/route_update",
                    "params": {
                        "itemId": int(resource_id),
                        "orders": route_orders,
                        "uid": route_id,
                        "callMode": "create",
                        "exp": 3600,
                        "f": 1,
                        "n": f"{vehicle_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "summary": {
                            "countOrders": len(route_orders),
                            "duration": route_summary['duration'] if route_summary and 'duration' in route_summary else 0,
                            "mileage": total_mileage,
                            "priceMileage": float(total_mileage) / 1000,
                            "priceTotal": total_cost,
                            "weight": total_weight,
                            "cost": total_cost
                        }
                    }
                }],
                "flags": 0
            }),
            "sid": session_id
        }

        st.info("Creating final route...")
        route_response = requests.post(base_url, data=batch_payload)
        route_result = route_response.json()

        if isinstance(route_result, list):
            has_error = any(isinstance(item, dict) and item.get('error', 0) != 0 for item in route_result)
            if not has_error:
                planning_url = (
                    f"https://apps.wialon.com/logistics/?"
                    f"lang=en&"
                    f"sid={token}#"
                    f"/distrib/step3"
                )
                return {
                    "error": 0,
                    "message": "Route created successfully",
                    "planning_url": planning_url,
                    "optimize_result": optimize_result,
                    "route_result": route_result
                }
            else:
                error_item = next((item for item in route_result if isinstance(item, dict) and item.get('error', 0) != 0), None)
                return {
                    "error": error_item.get('error', 1) if error_item else 1,
                    "message": error_item.get('reason', 'Unknown error') if error_item else 'Unknown error in batch response'
                }
        elif isinstance(route_result, dict):
            if route_result.get("error", 1) == 0:
                planning_url = (
                    f"https://apps.wialon.com/logistics/?"
                    f"lang=en&"
                    f"sid={token}#"
                    f"/distrib/step3"
                )
                return {
                    "error": 0,
                    "message": "Route created successfully",
                    "planning_url": planning_url,
                    "optimize_result": optimize_result,
                    "route_result": route_result
                }
            else:
                return route_result
        else:
            return {
                "error": 1,
                "message": f"Unexpected response type: {type(route_result)}"
            }

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Error details:", str(e))
        st.write("Error location:", e.__traceback__.tb_lineno)
        return {"error": 1, "message": f"An unexpected error occurred: {str(e)}"}


def run_wialon_uploader():
    st.subheader("\U0001F4E6 Logistics PDF Orders Uploader (via Logistics API)")

    with st.form("upload_form"):
        pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])
        excel_file = st.file_uploader("Upload Excel File (Assets)", type=["xls", "xlsx"])
        selected_date = st.date_input("Select Route Date")

        col1, col2 = st.columns(2)
        with col1:
            start_hour = st.slider("Route Start Hour", 0, 23, 6)
        with col2:
            end_hour = st.slider("Route End Hour", start_hour + 1, 23, 18)

        token = st.text_input("Enter your Wialon Token", type="password")
        resource_id = st.text_input("Enter Wialon Resource ID")

        submit_btn = st.form_submit_button("Upload and Dispatch")

    if submit_btn:
        if not pdf_file or not excel_file or not token or not resource_id:
            st.error("Please upload both files, and enter token and resource ID.")
        else:
            try:
                with st.spinner("Processing..."):
                    # Create datetime objects with explicit timezone handling
                    nairobi_tz = pytz.timezone('Africa/Nairobi')
                    
                    # Create the start and end times in Nairobi timezone
                    start_time = datetime.combine(selected_date, datetime.min.time().replace(hour=start_hour))
                    end_time = datetime.combine(selected_date, datetime.min.time().replace(hour=end_hour))
                    
                    # Localize to Nairobi timezone
                    start_time_local = nairobi_tz.localize(start_time)
                    end_time_local = nairobi_tz.localize(end_time)
                    
                    # Convert to UTC timestamps
                    tf = int(start_time_local.timestamp())
                    tt = int(end_time_local.timestamp())

                    # Debug information
                    st.write("Debug Time Info (please check these times):")
                    st.write(f"Start Time (Nairobi): {start_time_local.strftime('%Y-%m-%d %H:%M %Z')}")
                    st.write(f"End Time (Nairobi): {end_time_local.strftime('%Y-%m-%d %H:%M %Z')}")
                    st.write(f"Start Time (UTC): {datetime.utcfromtimestamp(tf).strftime('%Y-%m-%d %H:%M')} UTC")
                    st.write(f"End Time (UTC): {datetime.utcfromtimestamp(tt).strftime('%Y-%m-%d %H:%M')} UTC")
                    st.write(f"Timestamps - tf: {tf}, tt: {tt}")

                    gdf_joined, truck_number = read_pdf_to_df(pdf_file)
                    if gdf_joined is None:
                        return

                    unit_id, vehicle_name = read_asset_id_from_excel(excel_file, truck_number)
                    if not unit_id:
                        st.error(f"Could not find unit ID for truck: {truck_number}")
                        return

                    result = send_orders_and_create_route(token, int(resource_id), unit_id, vehicle_name, gdf_joined, tf, tt)

                    if result.get("error") == 0:
                        st.success("✅ Route created successfully!")
                        st.balloons()
                    else:
                        st.error(f"❌ Failed: {result.get('message', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_wialon_uploader()


