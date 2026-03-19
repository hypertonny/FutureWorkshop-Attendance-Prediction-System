# Quick Commands Reference

## 🚀 Deployment

```bash
# Easy deployment (recommended for first-time setup)
./deploy.sh

# Manual deployment
docker compose up -d --build
```

## 📊 Monitoring

```bash
# View all logs (live)
docker compose logs -f

# View specific service logs
docker compose logs -f streamlit
docker compose logs -f nginx

# Check container status
docker compose ps

# Check resource usage
docker stats
```

## 🔧 Management

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart streamlit

# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v

# Rebuild after code changes
docker compose up -d --build
```

## 🐛 Debugging

```bash
# Access container shell
docker compose exec streamlit bash
docker compose exec nginx sh

# Test Streamlit directly
docker compose exec streamlit curl http://localhost:8501/_stcore/health

# Check nginx configuration
docker compose exec nginx nginx -t

# View nginx error logs
docker compose logs nginx | grep error
```

## 🔄 Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build

# Or use deploy script
./deploy.sh
```

## 🌐 Access Points

- **Public**: https://model.bnhverse.tech (via Cloudflare)
- **Local**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health

## 🔒 Backup

```bash
# Backup data and models
tar -czf backup-$(date +%Y%m%d).tar.gz data/ models/ master_dataset.csv

# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz
```

## 🛑 Emergency

```bash
# Stop everything immediately
docker compose down

# Force remove all containers
docker compose down --remove-orphans

# Clean up Docker system (⚠️ removes unused images/containers)
docker system prune -a
```
