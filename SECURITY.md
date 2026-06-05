# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 1.0.x (latest) | ✅ Active |
| < 1.0 | ❌ No longer supported |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability — including issues with API authentication,
credential handling, dependency CVEs, or data exposure — please report it privately:

**Email:** [ahmedabbas52233@gmail.com](mailto:ahmedabbas52233@gmail.com)

**Subject line:** `[TerraVision AI] Security Vulnerability Report`

**Include in your report:**

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any relevant logs or screenshots (with sensitive data redacted)
- Your suggested fix, if you have one

---

## Response Timeline

| Stage | Target |
| --- | --- |
| Initial acknowledgement | Within 48 hours |
| Severity assessment | Within 5 business days |
| Fix or mitigation | Within 30 days for critical; 90 days for non-critical |
| Public disclosure | After fix is released |

---

## Security Best Practices for Deployment

When deploying TerraVision AI, please follow these guidelines:

- Set `TERRAVISION_ENV=production` — this restricts CORS to your configured origins
- Generate a strong `TERRAVISION_API_KEY` with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Never commit your `GCP_SERVICE_ACCOUNT_JSON` or `.streamlit/secrets.toml` to version control
- The `.gitignore` already excludes `secrets.toml` and `.env` — do not override this
- Review `CORS_ORIGINS` and set it to only the domains that should access your API
- Use the multi-stage Dockerfile provided — it runs as a non-root user

---

## Known Limitations

- The model checkpoint (`terravision_v1.pth`) is trained on synthetic data.
  It should not be used for operational agricultural or financial decisions
  without independent field validation.

- Prediction outputs are for research and educational purposes only.
EOF
echo "SECURITY.md created ✅"