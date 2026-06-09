# DataForge — Future Feature Roadmap

> Organizational features planned for enterprise and team-scale ML pipelines.

## 1. Pipeline Templates & Recipes

Save a full pipeline config (column mapping, cleaning steps, format, model params) as a reusable **recipe** (JSON/YAML). Apply it to new datasets with one click.

- Versioned recipe storage
- Share recipes across team members
- Public recipe marketplace

## 2. Multi-Dataset & Batch Processing

Handle multiple files simultaneously — upload a folder of CSVs, run the same pipeline on all, compare results.

- Batch upload (ZIP, folder)
- Batch mode: same recipe applied to N files
- Side-by-side profiling comparison
- Merge/union datasets before training

## 3. Experiment Tracking (DataForge MLflow)

Track every training run: hyperparameters, metrics, dataset hash, model artifact, timestamp.

- One-click compare across experiments
- Parameter importance analysis
- Visual run comparison (metric curves, confusion matrices side-by-side)
- Export to MLflow server

## 4. Model Registry

Version-managed model storage with staging/production gates.

- Register model versions with metadata
- Promote/demote between staging → production
- Model lineage: which dataset + recipe produced this model
- One-click rollback to previous version

## 5. Data Versioning & Lineage

Track every transformation applied to a dataset.

- Row-level provenance: which cleaning step changed which values
- Diffs between dataset versions
- Snapshot and restore any past state
- Checksum-based dedup storage

## 6. Deploy as REST API

Turn any trained model into a live HTTP endpoint (FastAPI + Docker).

- Auto-generated FastAPI app
- `/predict` endpoint with JSON input/output
- Swagger docs auto-generated
- One-click deploy to Hugging Face Spaces / Railway / Render
- Batch predict endpoint

## 7. Scheduled Automation

Run pipelines on a schedule without manual intervention.

- Cron-based scheduling of full pipelines
- Auto-import from S3 / GCS / Dropbox / email
- Auto-retrain on new data
- Slack / email / webhook notifications on completion

## 8. Multi-User & RBAC

Simple auth layer for team usage.

- Login with Google / GitHub OAuth
- Role-based access: Admin, Editor, Viewer
- Isolated workspaces per team/project
- Activity audit log

## 9. Data Source Integrations

Connect directly to databases and data warehouses.

- PostgreSQL, MySQL, SQLite
- Snowflake, BigQuery, Redshift, Databricks
- S3 / GCS / Azure Blob
- Google Sheets, Airtable, Notion
- Real-time streaming (Kafka, Kinesis)

## 10. Collaborative Labeling & Annotations

Built-in annotation UI for classification/sequence labeling tasks.

- Text classification annotation
- Multi-class, multi-label support
- Reviewer workflow (annotate → review → approve)
- Export annotations in standard formats (COCO, JSONL, ConLL)

## 11. Model Monitoring & Drift Detection

Track production model health over time.

- PSI (Population Stability Index) for data drift
- Performance decay alerts
- Scheduled re-evaluation against holdout set
- Dashboard: accuracy, latency, throughput over time

## 12. A/B Testing Framework

Compare model variants in production.

- Route traffic between model versions
- Track business metrics per variant
- Automated winner promotion

## 13. ONNX / TFLite Export

Export trained models for edge deployment.

- Convert sklearn pipelines to ONNX
- Optimize for mobile (TFLite / CoreML)
- Download as portable format + inference code

## 14. Pipeline DAG Visualizer

Interactive visualization of the full pipeline graph.

- Node: upload → clean → format → train → deploy
- Click any node to inspect config & output
- Highlight performance bottlenecks

## 15. Notification & Alert System

Get notified when key events happen.

- Training complete / failed
- Data drift detected
- New data available for retraining
- Channels: Slack, Discord, Email, Webhook

## 16. Usage Analytics Dashboard

Track how the team uses DataForge.

- Number of pipelines run per day/week
- Most popular cleaning ops
- Model training success rate
- Active users and projects

## 17. Export / Import Everything

Full portability between instances.

- Export project as ZIP (config + data + model)
- Import to another DataForge instance
- YAML/JSON config export for CI/CD

## 18. Terraform / Pulumi Provider

Infrastructure-as-code for DataForge pipelines.

- Define pipelines as Terraform resources
- GitOps workflow: PR → plan → apply
- Version-controlled pipeline definitions

## 19. Custom Plugin System

Extend DataForge with community or in-house plugins.

- Plugin marketplace
- Custom cleaning operations
- Custom model architectures
- Custom visualization widgets

## 20. Edge Case & Stress Testing Suite

Automated evaluation against adversarial inputs.

- Missing values, outliers, corrupted data
- Distribution shift simulation
- Model robustness scoring
- Generate report with improvement suggestions
