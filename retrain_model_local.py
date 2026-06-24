import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

print("Loading dataset...")

df = pd.read_csv("crop_recommendation_with_moisture.csv")

# Encode soil texture
texture_encoder = LabelEncoder()
df["soil_texture_encoded"] = texture_encoder.fit_transform(df["soil_texture"])

# Prepare features
feature_columns = [
    "N", "P", "K",
    "temperature", "humidity", "ph", "rainfall",
    "organic_matter_content", "crop_cycle_duration",
    "soil_texture_encoded",
    "soil_moisture"
]

X = df[feature_columns]
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y_encoded)

# Save model & encoders
joblib.dump(model, "crop_top3_model_new.joblib")
joblib.dump(label_encoder, "crop_label_encoder_new.joblib")
joblib.dump(texture_encoder, "soil_texture_encoder.joblib")

# Save feature order
with open("feature_columns_new.json", "w") as f:
    json.dump(feature_columns, f)

print("\nModel training complete!")
print("Saved files:")
print(" - crop_top3_model_new.joblib")
print(" - crop_label_encoder_new.joblib")
print(" - soil_texture_encoder.joblib")
print(" - feature_columns_new.json")
