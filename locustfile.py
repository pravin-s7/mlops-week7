from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        headers = {"Content-Type": "application/json"}
        payload = {
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2
        }
        self.client.post("/predict/", json=payload, headers=headers)