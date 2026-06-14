import json
import pathlib

# Find your downloaded JSON key automatically
json_files = list(pathlib.Path(".").glob("terravision-ai-*.json"))
if not json_files:
    print("❌ No terravision-ai-*.json file found. Put your key in this folder.")
    exit(1)

# Read and convert
with open(json_files[0]) as f:
    data = json.load(f)

# This produces the PERFECT single-line format
single_line = json.dumps(data, separators=(",", ":"))

# Print for Streamlit Cloud
print("=" * 60)
print("COPY THIS EXACT LINE — paste into Streamlit Cloud → Settings → Secrets:")
print("=" * 60)
print(f"GCP_SERVICE_ACCOUNT = '{single_line}'")
print("=" * 60)

# Also create local file
pathlib.Path(".streamlit").mkdir(exist_ok=True)
with open(".streamlit/secrets.toml", "w") as f:
    f.write(f"GCP_SERVICE_ACCOUNT = '{single_line}'\n")

print("\n✅ Also saved to .streamlit/secrets.toml for local use.")
