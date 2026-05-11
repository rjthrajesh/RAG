# AWS EC2 Deployment Guide

## Instance Recommendations

| Use Case | Instance | vCPU | RAM | Notes |
|---|---|---|---|---|
| Minimum (llama3.1:8b) | t3.xlarge | 4 | 16 GB | CPU inference only |
| Recommended (llama3.1:8b) | t3.2xlarge | 8 | 32 GB | More headroom |
| GPU (llama3.1:70b) | g4dn.xlarge | 4 | 16 GB + GPU | Much faster inference |

Storage: 50 GB EBS (models ~5 GB each, chroma + BM25 index ~1 GB).

## Setup Steps

```bash
# 1. Install Docker on Amazon Linux 2023
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Install Docker Compose plugin
sudo dnf install -y docker-compose-plugin

# 2. Clone repo and configure
git clone https://github.com/<your-org>/ask-my-docs.git
cd ask-my-docs
cp .env.example .env
# Edit .env with your values

# 3. Pull Ollama model (takes several minutes)
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b

# 4. Start the full stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Security Groups

| Rule | Port | Source |
|---|---|---|
| Allow inbound HTTP | 80 | 0.0.0.0/0 |
| Allow inbound HTTPS | 443 | 0.0.0.0/0 |
| Allow inbound SSH | 22 | Your IP only |
| Block | 8000, 8001, 11434 | Deny all (internal only) |

## Domain + SSL

```bash
# Install Certbot
sudo dnf install -y certbot

# Stop nginx temporarily for standalone challenge
docker compose stop nginx
sudo certbot certonly --standalone -d yourdomain.com

# Copy certs to repo certs/ directory
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/

# Update nginx.conf to add HTTPS listener and redirect
docker compose start nginx
```

## Monitoring

```bash
# Watch backend logs
docker compose logs -f backend

# Check health endpoint (use with UptimeRobot free tier)
curl http://yourdomain.com/api/health
# Expected: { "status": "ok", "ollama": true, "chroma": true }
```
