# metrics
## Description
Here we store the metrics that we use to evaluate the performance of our models, usally in the form of a log file.
The folder structure is as follows:
```
metrics
├── mlruns                  # all the mlflow runs should be stored here
├── lightning_logs          # all the pytorch lightning logs should be stored here
├── search_logs             # all the results from the DARTS should be stored here

```
In the future we might want to have a dedicated server for storing the logs, but for now we can just store them locally.

---

