import streamlit as st
import scrap_data
import process_data as preprocess
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        score = self.V(torch.tanh(self.W(x)))  # (B, T, 1)
        weights = torch.softmax(score, dim=1)  # (B, T, 1)
        context = torch.sum(weights * x, dim=1)  # (B, H)
        return context


# GRU + Embedding + Attention
class GRUFareModel(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=16, gru_units=112, dropout_rate=0.3):
        super().__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim) for v in vocab_sizes])

        # GRU stack for sequence input
        self.gru = nn.GRU(input_size=2, hidden_size=gru_units,
                          num_layers=2, batch_first=True, dropout=dropout_rate)
        self.attn = TemporalAttention(gru_units)

        # Dense layers for combined features
        input_static_dim = len(vocab_sizes) * embed_dim + 9  # 9 continuous features
        self.fc = nn.Sequential(
            nn.Linear(input_static_dim + gru_units, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_cont, x_seq):
        # Embedding branch
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_static = torch.cat(emb_list + [x_cont], dim=1)

        # GRU + attention branch
        gru_out, _ = self.gru(x_seq)  # (B, T, H)
        seq_ctx = self.attn(gru_out)  # (B, H)

        # Fuse
        fused = torch.cat([x_static, seq_ctx], dim=1)
        return self.fc(fused)


# Inject Bootstrap 5
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Navbar Header ---
# === MAIN CONTAINER ===
st.markdown("""<div class="container mt-4 p-0">""", unsafe_allow_html=True)
st.markdown("""
<nav class="navbar navbar-expand-lg rounded mb-4" style="background-color: #e3f2fd;">
  <div class="container-fluid">
        <h2><strong>Dashboard</strong></h2>
        <h5 class="text-center">Compair Absolute error and percentage error from  Actual Price and Predicted Price</h5>
  </div>
</nav>
""", unsafe_allow_html=True)

# --- Search Form ---

with st.form("flight_search"):
    st.markdown("### ✈️ Flight Search")
    col1, col2, col3, col4, col5, col6 = st.columns([1,3,3,1,2, 2])
    class_map = {
        'e': 'Economy',
        'w': 'Premium Economy',
        'b': 'Business'
    }
    airport_fullname = {
        "BOM": "Chhatrapati Shivaji Maharaj International Airport, Mumbai",
        "DEL": "Indira Gandhi International Airport, Delhi",
        "BLR": "Kempegowda International Airport, Bengaluru",
        "HYD": "Rajiv Gandhi International Airport, Hyderabad",
        "CCU": "Netaji Subhas Chandra Bose International Airport, Kolkata",
        "MAA": "Chennai International Airport, Chennai",
        "AMD": "Sardar Vallabhbhai Patel International Airport, Ahmedabad",
        "PNQ": "Pune International Airport, Pune",
        "GOI": "Dabolim Airport, Goa",
        "COK": "Cochin International Airport, Kochi"
    }
    with col1:
        trip_type = st.selectbox("Trip Type", ["One Way", "Round Trip"])
    with col2:
        origin = st.selectbox("From", options= list(airport_fullname.keys()), format_func= lambda x: airport_fullname[x] )
    with col3:
        destination = st.selectbox("To", options= list(airport_fullname.keys()), format_func= lambda x: airport_fullname[x] )
    with col4:
        depart_date = st.date_input("Depart", datetime.date.today())
    with col5:
        cubin =  st.selectbox(
            "Travel Class",
            options=list(class_map.keys()),  # ['e', 'w', 'b']
            format_func=lambda x: class_map[x]  # Show label instead of key
        )
    with col6:
        submit = st.form_submit_button("Search", use_container_width=True)


# --- Show Results only after Submit ---
if submit:
    st.subheader(f"Flights {origin} → {destination} on {depart_date.strftime('%d-%m-%Y')} {class_map[cubin]}")
    df = scrap_data.fetch_data_from_source(origin, destination, depart_date.strftime('%d%m%Y'), cubin)
    print(df.head())
    df_copy = df.copy()
    df = preprocess.preprocess_of_data(df)

    price_lookup = {
        (route, dep_date.strftime("%Y-%m-%d"), search.strftime("%Y-%m-%d")): price
        for route, dep_date, search, price in zip(
            df["RouteKey"],
            df["DepartureDate"],
            df["SearchDate"],
            df["Price"]
        )
    }

    actual_price = df["Price"].values

    # Encode category feature
    cat_cols = ["Airline", "Source", "Destination", "Class", "departure_segment", "arrival_segment"]
    vocab = {col: df[col].astype(str).unique().tolist() for col in cat_cols}
    tokenizers = {col: {v: i for i, v in enumerate(vocab[col])} for col in cat_cols}

    def encode_categorical(row):
        return [tokenizers[col][str(row[col])] for col in cat_cols]


    X_cat = np.array(df.apply(encode_categorical, axis=1).tolist())


    # start seq
    def build_sequence_fast(route_key, departure_date, look_back=7):
        seq = []
        for i in range(look_back, 0, -1):
            search_date = departure_date - datetime.timedelta(days=i)
            key = (route_key, departure_date.strftime("%Y-%m-%d"), search_date.strftime("%Y-%m-%d"))
            price = price_lookup.get(key, 0)
            seq.append([i, price])
        return seq


    seq_inputs = []
    look_back = 10
    X_seq = np.array([
        build_sequence_fast(row["RouteKey"], row["DepartureDate"])
        for _, row in df.iterrows()
    ], dtype=np.float32)

    X_cont_initial = df[['total_stop', 'search_date_day', 'search_date_month', 'days_until_departure', 'duration_mins',
                         'festival_season', 'dept_day_of_week', 'dept_day_of_year', 'dept_is_weekend']]
    y_initial = df[['Price']]

    # Train-test split
    # X_train, X_test, y_train, y_test, date_train, date_test, X_cat_train, X_cat_test, X_seq_train, X_seq_test, actual_train, actual_test = (
    #     train_test_split(X_cont_initial, y_initial, df['DepartureDate'], X_cat, X_seq, actual_price, test_size=0.2, random_state=42))

    # ===== Scale continuous X and target y
    with open("model/scalar_x.pkl", "rb") as f:
        scalar_X = pickle.load(f)
    scalar_X = StandardScaler()

    with open("model/scalar_y_new.pkl", "rb") as f:
        scalar_y = pickle.load(f)
    scalar_y = StandardScaler()

    X_test_scale = scalar_X.fit_transform(X_cont_initial)
    y_test_scale = scalar_y.fit_transform(y_initial.values.astype(np.float32))

    # Define vocab sizes exactly as during training
    with open("model/vocab_sizes.pkl", "rb") as f:
        vocab_sizes = pickle.load(f)
    print(vocab_sizes)
    # Rebuild architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUFareModel(vocab_sizes=vocab_sizes).to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load("model/gru_fare_model_new.pth", map_location=device))
    model.eval()

    print("✅ Model loaded successfully")

    # Pick random 60 indices
    idx = np.random.choice(len(X_test_scale), 60, replace=True)

    # Convert inputs to tensors
    x_cat_batch = torch.tensor(X_cat[idx], dtype=torch.long).to(device)
    x_cont_batch = torch.tensor(X_test_scale[idx], dtype=torch.float32).to(device)
    x_seq_batch = torch.tensor(X_seq[idx], dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        y_pred_sc = model(x_cat_batch, x_cont_batch, x_seq_batch).cpu().numpy()

    # Inverse scale
    y_pred = scalar_y.inverse_transform(y_pred_sc).flatten()
    y_true = y_initial.iloc[idx].values.flatten()  # 🔑 flatten ensures 1D

    # Build DataFrame
    ref_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })
    ref_df["Predicted_Price"] = ref_df["Predicted"].round(2)
    # Calculate error metrics for each row
    ref_df["Absolute_Error"] = np.abs(ref_df["Actual"] - ref_df["Predicted"]).round(2)
    ref_df["Percentage_Error"] = (
            100 * ref_df["Absolute_Error"] / ref_df["Actual"].replace(0, np.nan)
    ).round(2)

    # Save to CSV

    # ref_df.to_csv("sample_predictions.csv", index=False)
    # print("✅ Saved 60 random predictions to sample_predictions.csv")
    # print(ref_df.head())

    st.markdown("### 🧳 Travel Class Codes")
    # Bootstrap styled table
    st.markdown("""
       <div class="card mb-3 shadow-sm">
          <div class="card-body d-flex justify-content-between align-items-center">
               <div>Airlines</div>
               <div>Depart, Arrival / Duration / </div>
               <div>Price (₹)</div>
               <div>Predicted Price (₹)</div>
               <div>Absolute Error</div>
               <div>Percentage Error</div>
          </div>
        </div>
       </div>
    """, unsafe_allow_html=True)

    for i, (row1, row2) in enumerate(zip(df_copy.iterrows(), ref_df.iterrows()), start=1):
            idx1 , data1 = row1
            idx2, data2 = row2
            st.markdown(
                f"""
                <div class="card mb-3 shadow-sm">
                  <div class="card-body d-flex justify-content-between align-items-center">
                    <div>
                      <h6 class="card-title mb-0">{data1['Airline']}</h6>
                      <small class="text-muted">{data1['Stop']}</small>
                    </div>
                    <div>
                      <strong>{data1['Departure']}</strong> → <strong>{data1['Arrival']}</strong><br>
                      <small class="text-muted">{data1['Duration']}</small>
                    </div>
                    <div class="fw-bold text-success">{data2['Actual']}</div>
                <div class="fw-bold text-primary">{data2['Predicted']:.2f}</div>
                <div class="fw-bold text-danger">{data2['Absolute_Error']}</div>
                <div class="fw-bold text-warning">{data2['Percentage_Error']}%</div>
                   
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
    # --- Close Main Container ---
    st.markdown("</div>", unsafe_allow_html=True)