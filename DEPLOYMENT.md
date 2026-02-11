# 🐳 Docker Deployment Guide

## Prerequisites

Before deploying, ensure you have:
- ✅ Docker installed on your server
- ✅ Docker Compose installed
- ✅ Domain `model.bnhverse.tech` pointing to your server IP (140.238.231.167)
- ✅ Ports 80 and 443 open on your server firewall

## 📋 Quick Start

### 1. Install Docker & Docker Compose (if not already installed)

```bash
# Update package list
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Add your user to docker group (optional - to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### 2. Build and Run the Application

```bash
# Navigate to project directory
cd /home/ubuntu/FutureWorkshop-Attendance-Prediction-System

# Build and start containers
docker compose up -d --build

# Check if containers are running
docker compose ps

# View logs
docker compose logs -f
```

### 3. Access Your Application

Since your domain is proxied through Cloudflare:
- **URL**: http://model.bnhverse.tech
- Cloudflare will handle SSL/TLS encryption
- The app will be accessible over HTTPS automatically through Cloudflare

## 🔧 Configuration Notes

### Cloudflare SSL Mode

Since you're using Cloudflare proxy (orange cloud enabled):

1. **In Cloudflare Dashboard** (`model.bnhverse.tech`):
   - Go to SSL/TLS settings
   - Set SSL/TLS encryption mode to **"Flexible"** or **"Full"**
   - Flexible: Cloudflare ↔️ Browser (HTTPS), Cloudflare ↔️ Server (HTTP)
   - Full: Requires SSL on your server (see option below)

2. **Current Setup**: Configured for Flexible mode (HTTP only on server)

### If You Want nginx to Handle SSL (Optional)

If you prefer Full SSL mode:

1. Generate SSL certificates using Certbot:
```bash
# Install Certbot
sudo apt install certbot -y

# Stop nginx temporarily
docker compose stop nginx

# Generate certificate
sudo certbot certonly --standalone -d model.bnhverse.tech

# Certificates will be saved to:
# /etc/letsencrypt/live/model.bnhverse.tech/fullchain.pem
# /etc/letsencrypt/live/model.bnhverse.tech/privkey.pem
```

2. Copy certificates to nginx/ssl directory:
```bash
sudo cp /etc/letsencrypt/live/model.bnhverse.tech/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/model.bnhverse.tech/privkey.pem nginx/ssl/
sudo chmod 644 nginx/ssl/*.pem
```

3. Edit `nginx/conf.d/app.conf`:
   - Uncomment the HTTPS server block (lines starting with #)
   - Comment out or modify the HTTP server redirect line

4. Restart containers:
```bash
docker compose restart nginx
```

5. In Cloudflare, set SSL mode to **"Full"**

## 🎯 Docker Commands Cheat Sheet

```bash
# Start containers
docker compose up -d

# Stop containers
docker compose down

# Restart containers
docker compose restart

# View logs (all services)
docker compose logs -f

# View logs (specific service)
docker compose logs -f streamlit
docker compose logs -f nginx

# Rebuild after code changes
docker compose up -d --build

# Check container status
docker compose ps

# Execute command in running container
docker compose exec streamlit bash

# Remove everything (containers, networks, volumes)
docker compose down -v

# View resource usage
docker stats
```

## 🔍 Troubleshooting

### Container won't start
```bash
# Check logs for errors
docker compose logs streamlit
docker compose logs nginx

# Check if ports are already in use
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
```

### Can't access the application
```bash
# 1. Check if containers are running
docker compose ps

# 2. Check nginx configuration
docker compose exec nginx nginx -t

# 3. Verify Cloudflare DNS settings
# - Ensure model.bnhverse.tech points to 140.238.231.167
# - Check if Cloudflare proxy is enabled (orange cloud)

# 4. Check server firewall
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### Streamlit connection issues
```bash
# Check if Streamlit is responding
docker compose exec streamlit curl http://localhost:8501/_stcore/health

# Restart Streamlit container
docker compose restart streamlit
```

### Update application code
```bash
# After making code changes
docker compose up -d --build

# Or for just the app (faster)
docker compose build streamlit
docker compose up -d streamlit
```

## 📊 Monitoring

### Health Checks
```bash
# Streamlit health
curl http://localhost:8501/_stcore/health

# Or via nginx
curl http://model.bnhverse.tech/_stcore/health
```

### Logs Location
- Nginx logs: `./nginx/logs/`
- Container logs: `docker compose logs`

## 🔒 Security Recommendations

1. **Firewall**: Ensure only necessary ports are open
```bash
sudo ufw enable
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
```

2. **Regular Updates**: Keep Docker images updated
```bash
docker compose pull
docker compose up -d
```

3. **Backup Data**: Regularly backup your data and models
```bash
# Backup volumes
docker compose down
tar -czf backup-$(date +%Y%m%d).tar.gz data/ models/ master_dataset.csv
docker compose up -d
```

## 🚀 Production Checklist

- [ ] Docker and Docker Compose installed
- [ ] Application builds successfully (`docker compose build`)
- [ ] Containers start without errors (`docker compose up -d`)
- [ ] Application accessible at http://model.bnhverse.tech
- [ ] Cloudflare SSL/TLS mode configured (Flexible or Full)
- [ ] Firewall configured (ports 80, 443 open)
- [ ] Logs showing no errors (`docker compose logs`)
- [ ] Health check passing (`curl http://model.bnhverse.tech/_stcore/health`)

## 📝 Environment Variables (Optional)

If you need to add environment variables, create a `.env` file:

```bash
# .env file
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

Then reference it in `docker-compose.yml`:
```yaml
services:
  streamlit:
    env_file:
      - .env
```

## 🆘 Support

If you encounter issues:
1. Check logs: `docker compose logs -f`
2. Verify DNS: `nslookup model.bnhverse.tech`
3. Test locally: `curl http://localhost:8501`
4. Check Cloudflare settings in dashboard

---

**Your application is now containerized and ready for production! 🎉**
