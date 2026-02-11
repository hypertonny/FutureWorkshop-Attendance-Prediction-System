#!/bin/bash

# FutureWorkshop Attendance Prediction - Docker Setup Script
# This script automates the deployment process

set -e

echo "🚀 FutureWorkshop Attendance Prediction - Docker Deployment"
echo "=========================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
echo -e "${YELLOW}Checking prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: sudo apt install docker-compose-plugin -y"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
echo ""

# Check if ports are available
echo -e "${YELLOW}Checking if ports 80 and 443 are available...${NC}"
if sudo lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 80 is already in use${NC}"
    echo "You may need to stop the service using port 80 first"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✅ Ports check passed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p data models nginx/logs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Stop existing containers if running
echo -e "${YELLOW}Stopping existing containers (if any)...${NC}"
docker compose down 2>/dev/null || true
echo ""

# Build and start containers
echo -e "${YELLOW}Building Docker images...${NC}"
docker compose build

echo ""
echo -e "${YELLOW}Starting containers...${NC}"
docker compose up -d

echo ""
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if containers are running
if [ "$(docker compose ps -q streamlit)" ] && [ "$(docker compose ps -q nginx)" ]; then
    echo -e "${GREEN}✅ All containers are running${NC}"
    echo ""
    
    # Display container status
    echo -e "${YELLOW}Container Status:${NC}"
    docker compose ps
    echo ""
    
    # Test health check
    echo -e "${YELLOW}Testing application health...${NC}"
    sleep 5
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        echo -e "${GREEN}✅ Streamlit is healthy${NC}"
    else
        echo -e "${YELLOW}⚠️  Streamlit health check pending (this is normal on first start)${NC}"
    fi
    echo ""
    
    # Display success message
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}🎉 Deployment Successful!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo -e "Your application is now running at:"
    echo -e "${GREEN}🌐 http://model.bnhverse.tech${NC}"
    echo -e "   (via Cloudflare HTTPS proxy)"
    echo ""
    echo -e "Local access:"
    echo -e "   http://localhost:8501"
    echo ""
    echo -e "Useful commands:"
    echo -e "   ${YELLOW}View logs:${NC}       docker compose logs -f"
    echo -e "   ${YELLOW}Stop app:${NC}        docker compose down"
    echo -e "   ${YELLOW}Restart app:${NC}     docker compose restart"
    echo -e "   ${YELLOW}Rebuild app:${NC}     docker compose up -d --build"
    echo ""
    
else
    echo -e "${RED}❌ Some containers failed to start${NC}"
    echo -e "${YELLOW}Check logs with: docker compose logs${NC}"
    exit 1
fi
