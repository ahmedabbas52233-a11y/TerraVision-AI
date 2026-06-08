import json
import pathlib
import sys

def main():
    # Look for the JSON file automatically
    json_files = list(pathlib.Path('.').glob('terravision-ai-498310-15096810fcb5.json'))

    if not json_files:
        print("❌ No service account JSON found.")
        print("   Looked for: terravision-ai-498310-15096810fcb5.json")
        print("   Please put your downloaded .json file in this folder.")
        sys.exit(1)

    json_path = json_files[0]
    print(f"📄 Found: {json_path.name}")

    # Read and validate
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Your JSON file is corrupted: {e}")
        print("   Download a fresh key from Google Cloud Console.")
        sys.exit(1)

    # Check required fields
    required = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"❌ Missing fields: {missing}")
        sys.exit(1)

    print(f"✅ JSON valid | Project: {data['project_id']} | Email: {data['client_email']}")

    # Create .streamlit directory
    streamlit_dir = pathlib.Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)

    # Write secrets.toml with CORRECT formatting
    secrets_path = streamlit_dir / "secrets.toml"

    # json.dumps handles escaping properly
    json_string = json.dumps(data, separators=(',', ':'))

    # ✅ FIXED: Removed repr(). Use single quotes so \n stays as \n
    with open(secrets_path, 'w', encoding='utf-8') as f:
        f.write(f"GCP_SERVICE_ACCOUNT = '{json_string}'\n")

    print(f"\n✅ SUCCESS: Created {secrets_path}")
    print("\n📝 LOCAL (your computer):")
    print("   Run: streamlit run app.py")
    print("\n☁️  STREAMLIT CLOUD (share.streamlit.app):")
    print("   Copy this EXACT line and paste into Settings → Secrets:")
    print("-" * 60)
    print(f"GCP_SERVICE_ACCOUNT = '{json_string}'")
    print("-" * 60)
    print("\n⚠️  IMPORTANT: Delete any old text in the Secrets box before pasting.")

if __name__ == "__main__":
    main()