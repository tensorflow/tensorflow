# This workflow will build a docker container, publish it to Google Container Registry, and deploy it to GKE.
#
# To configure this workflow:
#
# 1. Ensure that your repository contains the necessary configuration for your Google Kubernetes Engine cluster, including deployment.yml, kustomization.yml, service.yml, etc.
#
# 2. Set up secrets in your workspace: GKE_PROJECT with the name of the project, GKE_EMAIL with the service account email, GKE_KEY with the service account key.
#
# 3. Change the values for the GKE_ZONE, GKE_CLUSTER and IMAGE environment variables (below).

name: Build and Deploy to GKE

on:
  push:
    branches:
    - master

# Environment variables available to all jobs and steps in this workflow
env:
  GKE_PROJECT: ${{ secrets.GKE_PROJECT }}
  GKE_EMAIL: ${{ secrets.GKE_EMAIL }}
  GITHUB_SHA: ${{ github.sha }}
  GKE_ZONE: us-west1-a
  GKE_CLUSTER: example-gke-cluster
  IMAGE: gke-test

jobs:
  setup-build-publish-deploy:
    name: Setup, Build, Publish, and Deploy
    runs-on: ubuntu-latest
    steps:

    - name: Checkout
      uses: actions/checkout@master

    # Setup gcloud CLI
    - uses: GoogleCloudPlatform/github-actions/setup-gcloud@master
      with:
        version: '270.0.0'
        service_account_email: ${{ secrets.GKE_EMAIL }}
        service_account_key: ${{ secrets.GKE_KEY }}

    # Configure docker to use the gcloud command-line tool as a credential helper
    - run: |
        # Set up docker to authenticate
        # via gcloud command-line tool.
        gcloud auth configure-docker
      
    # Build the Docker image
    - name: Build
      run: |        
        docker build -t gcr.io/"$GKE_PROJECT"/"$IMAGE":"$GITHUB_SHA" \
          --build-arg GITHUB_SHA="$GITHUB_SHA" \
          --build-arg GITHUB_REF="$GITHUB_REF" .

    # Push the Docker image to Google Container Registry
    - name: Publish
      run: |
        docker push gcr.io/$GKE_PROJECT/$IMAGE:$GITHUB_SHA
        
    # Set up kustomize
    - name: Set up Kustomize
      run: |
        curl -o kustomize --location https://github.com/kubernetes-sigs/kustomize/releases/download/v3.1.0/kustomize_3.1.0_linux_amd64
        chmod u+x ./kustomize

    # Deploy the Docker image to the GKE cluster
    - name: Deploy
      run: |
        gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE --project $GKE_PROJECT
        ./kustomize edit set image gcr.io/$GKE_PROJECT/$IMAGE:${GITHUB_SHA}
        ./kustomize build . | kubectl apply -f -
        kubectl rollout status deployment/gke-test
        kubectl get services -o wide
