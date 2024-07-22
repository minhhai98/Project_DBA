minikube start

---- start pro
.\helm.exe repo add prometheus-community https://prometheus-community.github.io/helm-charts
.\helm.exe install prometheus prometheus-community/prometheus --namespace monitoring
 kubectl expose service prometheus-server --type=NodePort --target-port=9090 --name=prometheus-server-np -n monitoring
kubectl get svc -n monitoring
minikube service prometheus-server-np -n monitoring

------ start grafana
.\helm repo add grafana https://grafana.github.io/helm-charts
.\helm install grafana grafana/grafana --namespace monitoring
kubectl expose service grafana --type=NodePort --target-port=3000 --name=grafana-np
kubectl get services -n monitoring
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
minikube service grafana -n monitoring


---------------------------------------------------
CÁCH THEO MILVUS HƯỚNG DẪN ------> Cách này chạy oke

# Cài milvus

helm repo add milvus https://zilliztech.github.io/milvus-helm/
Tải https://raw.githubusercontent.com/milvus-io/milvus-helm/master/charts/milvus/values.yaml
Nhớ đọc config trong nó

helm install my-release milvus/milvus -f standalone-values.yaml

# Prometheus
git clone https://github.com/prometheus-operator/kube-prometheus.git
cd kube-prometheus
kubectl apply --server-side -f manifests/setup
kubectl wait  --for condition=Established   --all CustomResourceDefinition   --namespace=monitoring
kubectl apply -f manifests/
kubectl patch clusterrole prometheus-k8s --type=json -p "$(echo '[{\"op\": \"add\", \"path\": \"/rules/-\", \"value\": {\"apiGroups\": [\"\"], \"resources\": [\"pods\", \"services\", \"endpoints\"], \"verbs\": [\"get\", \"watch\", \"list\"]}}]')"


# Đây là lệnh muốn delete các resource vừa tạo
kubectl delete --ignore-not-found=true -f manifests/ -f manifests/setup

# Bật service monitor của k8s cho milvus
.\helm.exe upgrade my-release milvus/milvus --set metrics.serviceMonitor.enabled=true --reuse-valuess

.\helm.exe upgrade my-release milvus/milvus --set attu.enabled=true --reuse-values
.\helm.exe upgrade my-release milvus/milvus --set grafana.enabled=true --reuse-values


# xuất port để vào dashboard
kubectl --namespace monitoring --address 0.0.0.0 port-forward svc/prometheus-k8s 9090
kubectl --namespace monitoring --address 0.0.0.0 port-forward svc/grafana 3000
kubectl --namespace default --address 0.0.0.0 port-forward svc/my-release-milvus-attu 3001:3000
kubectl --namespace default --address 0.0.0.0 port-forward svc/my-release-milvus 19530:19531
kubectl --namespace default --address 0.0.0.0 port-forward svc/my-release-minio  9001:9000


# Có 1 cách khác để xuất port
minikube service grafana -n monitoring
minikube service my-release-milvus-attu -n default
minikube service my-release-minio  -n monitoring


# Tạo datasource với url của prometheus là
http://prometheus-operated.monitoring.svc.cluster.local:9090

# Tải json này về để import mới dashboard
https://raw.githubusercontent.com/milvus-io/milvus/2.2.0/deployments/monitor/grafana/milvus-dashboard.json

