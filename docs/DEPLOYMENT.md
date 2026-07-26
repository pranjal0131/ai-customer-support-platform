# Deployment notes

Docker Compose is a local reference deployment, not a complete internet-facing configuration.

## Before production

- Put the frontend and API behind TLS and a trusted reverse proxy.
- Replace local database credentials; use a secret manager and a managed PostgreSQL service with backups and point-in-time recovery.
- Add Alembic migrations instead of relying on startup `create_all`.
- Add OIDC authentication, tenant-aware authorization, rate limiting, request-size limits, and CSRF controls where relevant.
- Disable permissive origins and set `CORS_ORIGINS` to exact trusted dashboard origins.
- Mount immutable, scanned model artifacts from a registry and record their version in every prediction audit event.
- Send JSON logs and traces to protected observability storage; never log raw ticket text by default.
- Add queue workers for slower transformer inference and autoscale separately from the API.
- Restrict database and model networks, run containers read-only where practical, and scan images/dependencies continuously.
- Establish retention/deletion policies, incident response, human escalation paths, and safe rollback procedures.

## Health and operations

`GET /health` reports process health, demo status, and the implementation active for each model component. Extend it with separate readiness checks for the database and artifact registry in an orchestrated deployment. Liveness should not depend on optional model providers.

## Artifact promotion

1. Train in an isolated reproducible environment.
2. Evaluate on immutable test and domain-review sets.
3. Review confusion matrices, calibration, safety slices, and license provenance.
4. Sign and register the artifact plus tokenizer/vectorizer and label mapping.
5. Deploy to a canary, monitor abstentions and agent overrides, then promote or roll back.
