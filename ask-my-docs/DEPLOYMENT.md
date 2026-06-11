# Deployment Guide

---

## Oracle Cloud A1 (Free Tier — Recommended)

Oracle Cloud's Always Free ARM64 A1 instances give you **4 OCPUs + 24 GB RAM at no cost** — plenty of headroom for the backend + ChromaDB without OOM.

The frontend stays on Vercel (free). The backend and ChromaDB run on OCI. The LLM runs on Groq (free tier).

### 1 — Provision the instance

In the OCI web console:
- **Shape**: VM.Standard.A1.Flex — 4 OCPU, 24 GB RAM
- **Image**: Canonical Ubuntu 22.04
- **Boot volume**: 50 GB (free up to 200 GB total)
- **SSH key**: upload your public key

After the instance is running, **open ports 80 and 443 in the Security List**:

> Networking → Virtual Cloud Networks → `<your VCN>` → Security Lists → Default Security List → Add Ingress Rules
>
> | Source CIDR | Protocol | Destination Port |
> |---|---|---|
> | 0.0.0.0/0 | TCP | 80 |
> | 0.0.0.0/0 | TCP | 443 |
> | `<your IP>` | TCP | 22 (SSH) |

### 2 — Set up the VM

```bash
ssh ubuntu@<oci-public-ip>

# Download and run the one-shot setup script (installs Docker, opens iptables ports)
bash <(curl -fsSL https://raw.githubusercontent.com/<your-org>/ask-my-docs/main/scripts/setup-oracle.sh)

# Log out and back in so the 'docker' group takes effect
exit
```

Or manually:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save
```

### 3 — Clone and configure

```bash
git clone https://github.com/<your-org>/ask-my-docs.git
cd ask-my-docs

# Copy your local .env or create one from scratch
# Required values:
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=gsk_...
#   CORS_ORIGINS=http://localhost:3000,https://<your-app>.vercel.app
nano .env
```

### 4 — Deploy

```bash
docker compose -f docker-compose.oci.yml up -d --build
```

The build runs **on the ARM64 VM** itself, so there is no cross-compilation. The first build downloads the ONNX embedding model and reranker (~500 MB) and bakes them into the image — subsequent restarts are instant.

### 5 — Point Vercel at the new backend

In your Vercel project dashboard → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL = http://<oci-public-ip>
```

Redeploy the Vercel project for the change to take effect.

### Useful commands

```bash
# Tail backend logs
docker compose -f docker-compose.oci.yml logs -f backend

# Health check
curl http://<oci-public-ip>/health

# Restart just the backend
docker compose -f docker-compose.oci.yml restart backend

# Pull latest code and rebuild
git pull
docker compose -f docker-compose.oci.yml up -d --build backend
```

### OCI security list vs. iptables

OCI has **two independent firewalls**: the VCN Security List (in the web console) and the OS-level iptables rules inside the instance. **Both must allow ports 80/443** — opening only one is a common gotcha.

---

## AWS EC2

### Instance Recommendations

| Use Case | Instance | vCPU | RAM | Notes |
|---|---|---|---|---|
| Minimum (llama3.1:8b) | t3.xlarge | 4 | 16 GB | CPU inference only |
| Recommended (llama3.1:8b) | t3.2xlarge | 8 | 32 GB | More headroom |
| GPU (llama3.1:70b) | g4dn.xlarge | 4 | 16 GB + GPU | Much faster inference |

Storage: 50 GB EBS (models ~5 GB each, chroma + BM25 index ~1 GB).

### Setup

```bash
# 1. Install Docker on Amazon Linux 2023
sudo dnf update -y
sudo dnf install -y docker docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

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

### Security Groups

| Rule | Port | Source |
|---|---|---|
| Allow inbound HTTP | 80 | 0.0.0.0/0 |
| Allow inbound HTTPS | 443 | 0.0.0.0/0 |
| Allow inbound SSH | 22 | Your IP only |
| Block | 8000, 8001, 11434 | Deny all (internal only) |

### Domain + SSL

```bash
sudo dnf install -y certbot
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop nginx
sudo certbot certonly --standalone -d yourdomain.com
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/
docker compose -f docker-compose.yml -f docker-compose.prod.yml start nginx
```

---

## Monitoring

```bash
# Health endpoint — wire to UptimeRobot free tier for uptime alerting
curl http://<host>/health
# Expected: { "status": "ok", "chroma": true }
```
