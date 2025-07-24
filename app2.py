# app_wialon.py
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

    required_columns = [
        "REP", "CUSTOMER ID", "CUSTOMER NAME", "LOCATION", "LOCATION COORDINATES",
        "INVOICE NO.", "AMOUNT", "TONNAGE"
    ]

    all_rows = []
    header = None

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
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

    header_keywords = ['REP', 'CUSTOMER ID', 'CUSTOMER NAME', 'INVOICE NO.', 'AMOUNT', 'TONNAGE']
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: any(str(cell).strip().upper() in header_keywords for cell in row), axis=1)]
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.astype(str).str.contains('Fixed|Driver Sign|Mileage|Cartons', case=False).any(), axis=1)]

    df_cleaned = df_cleaned[[col for col in df_cleaned.columns if col.strip().upper() in required_columns]]
    df_cleaned.columns = [col.strip().upper() for col in df_cleaned.columns]

    for col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()

    df_cleaned = df_cleaned.dropna(how='all')
    df_cleaned = df_cleaned[df_cleaned['CUSTOMER ID'].notna()].reset_index(drop=True)

    df_cleaned[['LAT', 'LONG']] = df_cleaned['LOCATION COORDINATES'].apply(lambda x: pd.Series(extract_coordinates(x)))
    df_cleaned = df_cleaned.dropna(subset=['LAT', 'LONG'])

    coords_rad = np.radians(df_cleaned[['LAT', 'LONG']])
    epsilon = 5 / 6371.0088
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    df_cleaned['Cluster'] = db.labels_

    df_cleaned = df_cleaned.sort_values(by=['Cluster', 'LAT', 'LONG']).reset_index(drop=True)

    gdf_points = gpd.GeoDataFrame(df_cleaned, geometry=[Point(xy) for xy in zip(df_cleaned['LONG'], df_cleaned['LAT'])], crs="EPSG:4326")
    kenya_counties = gpd.read_file("kenya-counties-simplified.geojson").to_crs("EPSG:4326")
    gdf_joined = gpd.sjoin(gdf_points, kenya_counties[['shapeName', 'geometry']], how="left", predicate="within")
    gdf_joined = gdf_joined.rename(columns={'shapeName': 'Correct County'})
    gdf_joined = gdf_joined.sort_values(by='Correct County').reset_index(drop=True)

    st.subheader("🗺️ Locations Matched with Counties")
    st.dataframe(gdf_joined[[
        'REP', 'CUSTOMER ID', 'CUSTOMER NAME', 'LOCATION', 'LOCATION COORDINATES',
        'INVOICE NO.', 'AMOUNT', 'TONNAGE', 'LAT', 'LONG', 'Correct County'
    ]])

    return gdf_joined

def convert_to_orders(df_cleaned, tf, tt):
    orders = []

    for idx, row in df_cleaned.iterrows():
        latitude, longitude = extract_coordinates(row.get('LOCATION COORDINATES', ''))
        if latitude is None or longitude is None:
            st.warning(f"Skipping row {idx} due to invalid coordinates.")
            continue

        try:
            weight_kg = int(float(row.get('TONNAGE', 0)) * 1000)
        except (ValueError, TypeError):
            weight_kg = 0

        order = {
            "itemId": 25601229,
            "id": 0,
            "n": row['CUSTOMER NAME'],
            "oldOrderId": 0,
            "oldOrderFiles": [],
            "p": {
                "n": row['CUSTOMER NAME'],
                "p": "", "p2": "", "e": "",
                "a": f"{row['Correct County']} (LAT: {latitude}, LONG: {longitude})",
                "v": 50, "w": weight_kg, "c": row['AMOUNT'], "ut": 1800,
                "t": "", "d": row.get('DELIVERY', 'No description'),
                "uic": str(row['INVOICE NO.']), "cid": str(row['CUSTOMER ID']),
                "cm": "Handle with care", "aff": "123,456", "z": "1001_2002",
                "ntf": 3, "pr": 1
            },
            "rp": "Enroute", "f": 0, "tf": tf, "tt": tt,
            "trt": 600, "r": 20, "y": latitude, "x": longitude,
            "ej": {}, "tz": 3,
            "cf": {"delivery_notes": "", "payment_status": ""},
            "callMode": "create", "dp": []
        }
        orders.append(order)

    return orders


def login_to_wialon(token):
    login_url = "https://hst-api.wialon.com/wialon/ajax.html"
    params = {"svc": "token/login", "params": json.dumps({"token": token})}
    response = requests.get(login_url, params=params)
    data = response.json()
    if 'eid' not in data:
        raise Exception(f"Login failed: {data}")
    return data['eid']

def send_orders(orders, sid):
    order_url = f"https://hst-api.wialon.com/wialon/ajax.html?sid={sid}"
    for order in orders:
        payload = {"svc": "order/update", "params": json.dumps(order)}
        response = requests.post(order_url, params=payload)
        result = response.json()
        st.success(f"Sent order: {order['n']}, Response: {result}")
        time.sleep(1)

def run_wialon_uploader():
    st.subheader("📦 Logistics PDF Orders Uploader")

    with st.form("upload_form"):
        pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])
        selected_date = st.date_input("Select Date")
        token = st.text_input("Enter your token", type="password")
        submit_btn = st.form_submit_button("Upload and Send Orders")

    if submit_btn:
        if not pdf_file or not token:
            st.error("Please upload a PDF and enter a token.")
        else:
            try:
                nairobi_tz = pytz.timezone("Africa/Nairobi")
                start_dt = nairobi_tz.localize(datetime.combine(selected_date, datetime.min.time()))
                end_dt = start_dt + timedelta(days=1)
                tf = int(start_dt.timestamp())
                tt = int(end_dt.timestamp())

                st.info("Processing PDF...")
                df_cleaned = read_pdf_to_df(pdf_file)
                st.success("PDF parsed successfully!")

                orders = convert_to_orders(df_cleaned, tf, tt)
                st.info(f"{len(orders)} valid orders ready.")

                sid = login_to_wialon(token)
                send_orders(orders, sid)

            except Exception as e:
                st.error(f"❌ Error: {e}")
