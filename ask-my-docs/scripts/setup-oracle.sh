#!/usr/bin/env bash
# One-shot setup for an Oracle Cloud A1 instance running Ubuntu 22.04.
# Run as the default 'ubuntu' user immediately after provisioning.
set -euo pipefail

# ── Docker ────────────────────────────────────────────────────────────────────
echo "→ Installing Docker..."
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker

# ── OS-level firewall (OCI drops all inbound by default via iptables) ─────────
echo "→ Opening ports 80 and 443..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save

# ── Git ───────────────────────────────────────────────────────────────────────
sudo apt-get install -y git

echo ""
echo "✓ Setup complete."
echo ""
echo "IMPORTANT: also open ports 80 and 443 in the OCI Security List (web console):"
echo "  Networking → Virtual Cloud Networks → <your VCN>"
echo "  → Security Lists → Default Security List → Add Ingress Rules"
echo "  Source: 0.0.0.0/0  IP Protocol: TCP  Destination Port: 80"
echo "  Source: 0.0.0.0/0  IP Protocol: TCP  Destination Port: 443"
echo ""
echo "Then log out, log back in (to pick up the docker group), and run:"
echo ""
echo "  git clone <your-repo> && cd ask-my-docs"
echo "  cp .env .env.bak"
echo "  # Edit .env: set LLM_PROVIDER=groq, GROQ_API_KEY=..., CORS_ORIGINS=..."
echo "  docker compose -f docker-compose.oci.yml up -d --build"
echo ""
echo "Tail logs:"
echo "  docker compose -f docker-compose.oci.yml logs -f backend"
