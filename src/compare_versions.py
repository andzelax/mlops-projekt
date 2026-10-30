from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "breast_cancer"

client = MlflowClient(tracking_uri=TRACKING_URI)

v1_meta = client.get_model_version(name=MODEL_NAME, version=1)
v2_meta = client.get_model_version(name=MODEL_NAME, version=2)

run_v1 = client.get_run(v1_meta.run_id)
run_v2 = client.get_run(v2_meta.run_id)

v1_acc = run_v1.data.metrics.get("accuracy", 0.0)
v1_f1 = run_v1.data.metrics.get("f1_macro", 0.0)

v2_acc = run_v2.data.metrics.get("accuracy", 0.0)
v2_f1 = run_v2.data.metrics.get("f1_macro", 0.0)

print("=" * 60)
print(f"PORÓWNANIE WERSJI MODELU: '{MODEL_NAME}'")
print("=" * 60)
print(f"Wersja 1 (Run ID: {v1_meta.run_id[:8]}...):")
print(f"  - Accuracy: {v1_acc:.4f}")
print(f"  - F1-macro: {v1_f1:.4f}")
print("-" * 60)
print(f"Wersja 2 (Run ID: {v2_meta.run_id[:8]}...):")
print(f"  - Accuracy: {v2_acc:.4f}")
print(f"  - F1-macro: {v2_f1:.4f}")
print("=" * 60)

if v2_f1 > v1_f1:
    better_version = 2
    better_f1 = v2_f1
else:
    better_version = 1
    better_f1 = v1_f1

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=better_version,
)

print(
    f"Decyzja: Wersja {better_version} osiągnęła wyższą/równą wartość F1-macro ({better_f1:.4f})."
)
print(f"Pomyślnie przepięto alias 'champion' na wersję {better_version}.")
