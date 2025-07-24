# app_tonnage.py
import streamlit as st
import pandas as pd

def run_tonnage_lookup():
    st.subheader("🚛 Vehicle Tonnage Lookup")
    
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file:
        try:
            raw_df = pd.read_excel(uploaded_file)

            columns_to_drop = [
                'NO.', 'Unnamed: 3', 'Unnamed: 6', 'Unnamed: 9', 'Unnamed: 12', 'Unnamed: 15',
                'Unnamed: 18', 'Unnamed: 21', 'Unnamed: 24', 'Unnamed: 27', 'Unnamed: 30',
                'Unnamed: 33', 'Unnamed: 36', 'Unnamed: 39', 'Unnamed: 42',
                'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 48', 'Unnamed: 49', 'TONNAGE.15',
                'Unnamed: 51', 'Unnamed: 52', 'TONNAGE.16', 'Unnamed: 54', 'Unnamed: 55',
                'TONNAGE.17'
            ]
            raw_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            data = []
            i = 0
            while i < len(raw_df.columns) - 1:
                region_col = raw_df.columns[i]
                tonnage_col = raw_df.columns[i + 1]

                if "TONNAGE" in str(tonnage_col).upper() and pd.notna(region_col):
                    for reg, ton in zip(raw_df[region_col], raw_df[tonnage_col]):
                        if pd.notna(reg) or pd.notna(ton):
                            data.append({
                                "Registration Number": reg if pd.notna(reg) else '',
                                "Region": region_col.strip(),
                                "Tonnage": ton if pd.notna(ton) else ''
                            })
                    i += 2
                else:
                    i += 1

            df = pd.DataFrame(data).fillna('')

            st.success("✅ File processed successfully!")

            st.sidebar.header("🔍 Filter Options")
            all_regions = sorted(df["Region"].unique())
            selected_region = st.sidebar.selectbox("Select Region (optional)", [""] + all_regions)

            filtered_df = df.copy()
            if selected_region:
                filtered_df = filtered_df[filtered_df["Region"] == selected_region]

            all_plates = sorted(df["Registration Number"].unique())
            selected_reg = st.sidebar.selectbox("Select Registration Number (optional)", [""] + all_plates)

            if selected_reg:
                reg_df = df[df["Registration Number"] == selected_reg]
                st.subheader(f"📋 Vehicle Details: {selected_reg}")
                st.dataframe(reg_df, use_container_width=True)
            elif selected_region:
                st.subheader(f"📍 All Vehicles in {selected_region}")
            else:
                st.subheader("📍 All Vehicles")

            def generate_tooltip_table(df):
                html = """
                <style>
                .tooltip {
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                    color: #1f77b4;
                    font-weight: bold;
                }
                .tooltip .tooltiptext {
                    visibility: hidden;
                    width: 220px;
                    background-color: black;
                    color: #fff;
                    text-align: left;
                    border-radius: 6px;
                    padding: 6px;
                    position: absolute;
                    z-index: 1;
                    bottom: 150%;
                    left: 50%;
                    transform: translateX(-50%);
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 14px;
                }
                .tooltip:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                th, td {
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    text-align: left;
                    font-size: 16px;
                }
                th {
                    background-color: #f4f4f4;
                }
                </style>
                <table>
                    <thead>
                        <tr>
                            <th>Registration Number</th>
                            <th>Region</th>
                            <th>Tonnage</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for _, row in df.iterrows():
                    reg = row['Registration Number']
                    region = row['Region']
                    tonnage = row['Tonnage']

                    tooltip = f"""
                    <div class='tooltip'>{reg}
                        <span class='tooltiptext'>
                            Region: {region if region else '—'}<br>
                            Tonnage: {tonnage if tonnage else '—'}
                        </span>
                    </div>
                    """
                    html += f"<tr><td>{tooltip}</td><td>{region}</td><td>{tonnage}</td></tr>"

                html += "</tbody></table>"
                return html

            st.markdown("### 🧾 Vehicles Table (hover over plate numbers):")
            display_df = df[df["Registration Number"] == selected_reg] if selected_reg else filtered_df
            st.markdown(generate_tooltip_table(display_df), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("📤 Please upload an Excel file to get started.")
