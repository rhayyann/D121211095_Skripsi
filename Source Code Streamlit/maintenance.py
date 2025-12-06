import streamlit as st
import pandas as pd
import io
import re
import os
from streamlit_agraph import agraph, Node, Edge, Config
from annotated_text import annotated_text

# --- Load Data from Files ---
try:
    vehicles_df = pd.read_csv("Data/vehicles.csv")
    maintenance_df = pd.read_csv("Data/maintenance_history.csv")
    kg_df = pd.read_csv("Data/maintenance_dashboard_kg.csv")
    # Prediction data is now loaded from the specified CSV file
    prediction_df = pd.read_csv("Data/sparepart_scores.csv")
    prediction_no_service_df = pd.read_csv("Data/sparepart_scores_no_service.csv")
except FileNotFoundError as e:
    st.error(f"Error loading data file: {e}. Make sure all CSV files are in the 'Data' directory.")
    st.stop()

# --- Status Categorization ---
def categorize_score(df: pd.DataFrame) -> pd.DataFrame:
    """Categorizes scores and assigns a status."""
    top_10 = df.nlargest(10, 'score')
    min_score = top_10['score'].min()
    max_score = top_10['score'].max()

    if max_score == min_score:
        # Avoid division by zero if all scores in the top 10 are the same
        df['score_norm'] = 0.5  # Assign a neutral normalized score
    else:
        df['score_norm'] = (df['score'] - min_score) / (max_score - min_score)

    def assign_status(score_norm):
        if score_norm >= 0.7:
            return ("Segera Cek", "#E6B0AA")  # Even Bolder Pastel Red
        elif 0.4 <= score_norm < 0.7:
            return ("Perlu Perhatian", "#FAD7A0")  # Even Bolder Pastel Yellow
        else:
            return ("Opsional", "#A9DFBF")  # Even Bolder Pastel Green

    df['status'] = df['score_norm'].apply(assign_status)
    return df

# --- Data Fetching Functions ---
def get_car_history(frame_serial_no: str) -> pd.DataFrame:
    """Fetches car maintenance history from the maintenance_df DataFrame."""
    history = maintenance_df[maintenance_df['Frame Serial No'] == frame_serial_no]
    return history

def get_part_prediction(frame_serial_no: str) -> pd.DataFrame:
    """Mock function to get spare part failure predictions for a given car."""
    # This function now returns the data loaded from sparepart_scores.csv
    if frame_serial_no in maintenance_df['Frame Serial No'].unique():
        return prediction_df, prediction_no_service_df
    return pd.DataFrame()

# --- Streamlit Dashboard UI ---

# --- Input Section ---
search_input = st.text_input(
    label="Search by Frame Serial Number or Vehicle Unit",
    placeholder="Enter Frame Serial No or License Plate...",
    label_visibility="collapsed"
)

# --- Display Area ---
selected_serial_no = None

if search_input:
    query = search_input.upper().strip()
    match = vehicles_df[vehicles_df['Frame Serial No'].str.upper() == query]

    if match.empty:
        vehicles_df['Normalized Vehicle Unit'] = vehicles_df['Vehicle Unit'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        normalized_query = re.sub(r'[^A-Z0-9]', '', query)
        match = vehicles_df[vehicles_df['Normalized Vehicle Unit'] == normalized_query]

    if not match.empty:
        selected_serial_no = match.iloc[0]['Frame Serial No']
    else:
        st.warning(f"No vehicle found for '{search_input}'. Please check your input.")

if selected_serial_no:
    car_history = get_car_history(selected_serial_no).copy()
    part_prediction, part_prediction_no_service = get_part_prediction(selected_serial_no)
    vehicle_info = vehicles_df[vehicles_df['Frame Serial No'] == selected_serial_no].iloc[0]

    # --- ROW 1: Vehicle Information ---
    with st.container(border=True):
        col_info, col_img = st.columns([2, 1])
        with col_info:
            st.markdown(f"""
<div style="line-height: 0.1; margin-bottom: 1rem;">
    <h2 style="margin: 0; padding: 0;">{vehicle_info['Vehicle Model']}</h2>
    <h4 style="color: green; margin: 0; padding: 0;">{vehicle_info['Vehicle Unit']}</h4>
</div>
""", unsafe_allow_html=True)
            # annotated_text((f"**{vehicle_info['Vehicle Unit']}**", "", "#008940"))
            if not car_history.empty:
                car_history['Work Order Date'] = pd.to_datetime(car_history['Work Order Date'], dayfirst=True, errors='coerce').dt.date
                latest_record = car_history.sort_values(by='Work Order Date', ascending=False).iloc[0]
                latest_mileage = int(latest_record['Current Mileage Record'])
                st.markdown(f"""
                Base Model : **{vehicle_info['Base Model']}**  
                Frame Serial No : **{vehicle_info['Frame Serial No']}**  
                Last Recorded Mileage : **{latest_mileage:,} km**
                """)
            else:
                st.markdown(f"""
                Base Model : **{vehicle_info['Base Model']}**  
                Frame Serial No : **{vehicle_info['Frame Serial No']}**  
                Last Recorded Mileage : **No maintenance record found**
                """)
        with col_img:
            base_model = vehicle_info['Base Model']
            image_path = os.path.join('elements', 'car-images', f'{base_model}.png')
            
            if os.path.exists(image_path):
                st.image(image_path)
            else:
                st.image("https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found")

    st.write("")

    if not car_history.empty:
        # --- ROW 2: Maintenance History ---
        with st.container():
            st.subheader("Riwayat Pemeliharaan")
            display_history = car_history[['Work Order', 'Work Order Date', 'Part Name', 'standarized_part_name', 'subsystem', 'Job', 'Current Mileage Record']]
            display_history = display_history.rename(columns={
                'Work Order Date': 'Tanggal', 
                'Part Name': 'Part Diganti', 
                'standarized_part_name': 'Nama Part', 
                'subsystem': 'Subsistem',
                'Current Mileage Record': 'Kilometer'
            }).sort_values(by='Tanggal', ascending=False).reset_index(drop=True)
            
            # Format Mileage with "km" suffix
            display_history['Kilometer'] = display_history['Kilometer'].apply(lambda x: "N/A km" if pd.isna(x) else f"{int(x):,} km")
            
            st.dataframe(display_history, use_container_width=True, height=250)
        st.write("")

        # --- Row 3: Knowledge Graph and Prediction ---
        col_kg, col_prediction = st.columns([3, 2])

        with col_kg:
            st.subheader("Graf Pemeliharaan")
            with st.container(border=True):
                # --- AGraph Implementation ---
                work_orders_df = kg_df[(kg_df['head'] == selected_serial_no)]
                relevant_work_orders = work_orders_df['tail'].tolist()
                parts_df = kg_df[kg_df['head'].isin(relevant_work_orders)]

                graph_data = pd.concat([work_orders_df, parts_df])

                nodes = []
                edges = []
                node_set = set()

                if not graph_data.empty:
                    for _, row in graph_data.iterrows():
                        head, relation, tail = row['head'], row['relation'], row['tail']

                        if head not in node_set:
                            node_set.add(head)
                            nodes.append(Node(id=head, label=head,
                                              color="#FFC72C" if relation == "HAS_WORK_ORDER" else "#008940",
                                              size=20 if relation == "HAS_WORK_ORDER" else 15))
                        if tail not in node_set:
                            node_set.add(tail)
                            nodes.append(Node(id=tail, label=tail,
                                              color="#008940" if relation == "HAS_WORK_ORDER" else ("#3E99EE" if relation == "INVOLVES_JOB" else "#D3D3D3"),
                                              size=15))

                        edges.append(Edge(source=head, target=tail, label=relation, font={'size': 5}))

                    config = Config(width=800,
                                    height=550,
                                    directed=True,
                                    physics=True,
                                    hierarchical=False,
                                    nodeHighlightBehavior=True,
                                    highlightColor="#F7A7A6",
                                    collapsible=True
                                    )

                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info("No maintenance graph data available for this vehicle.")

        with col_prediction:
            st.subheader("Saran Pemeliharaan")
            
            # --- Tabs for Predictions ---
            tab1, tab2 = st.tabs(["Semua Part", "Tanpa Part Berkala"])

            with tab1:
                with st.container(border=True):
                    display_prediction = part_prediction.copy().rename(columns={'item_name': 'Part Name'})
                    display_prediction = categorize_score(display_prediction)
                    
                    top_10_prediction = display_prediction.sort_values(by='score', ascending=False).head(10)

                    col1, col2, col3 = st.columns([3, 1, 2])
                    with col1: st.markdown("**Part Name**")
                    with col2: st.markdown("**Score**")
                    with col3: st.markdown("**Status**")

                    for _, row in top_10_prediction.iterrows():
                        part_name = row['Part Name']
                        score = f"{row['score']:.2f}"
                        status_text, status_color = row['status']
                        
                        col1, col2, col3 = st.columns([3, 1, 2])
                        with col1: st.markdown(part_name)
                        with col2: st.markdown(score)
                        with col3: annotated_text((status_text, "", status_color))

            with tab2:
                with st.container(border=True):
                    display_prediction_no_service = part_prediction_no_service.copy().rename(columns={'item_name': 'Part Name'})
                    display_prediction_no_service = categorize_score(display_prediction_no_service)
                    
                    top_10_prediction_no_service = display_prediction_no_service.sort_values(by='score', ascending=False).head(10)

                    col1, col2, col3 = st.columns([3, 1, 2])
                    with col1: st.markdown("**Part Name**")
                    with col2: st.markdown("**Score**")
                    with col3: st.markdown("**Status**")

                    for _, row in top_10_prediction_no_service.iterrows():
                        part_name = row['Part Name']
                        score = f"{row['score']:.2f}"
                        status_text, status_color = row['status']
                        
                        col1, col2, col3 = st.columns([3, 1, 2])
                        with col1: st.markdown(part_name)
                        with col2: st.markdown(score)
                        with col3: annotated_text((status_text, "", status_color))
    else:
        st.info(f"This vehicle exists in the system but has no maintenance history on record.")

elif not search_input:
    st.info("Please enter a Frame Serial Number or Vehicle Unit to begin.")