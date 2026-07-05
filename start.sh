export OLLAMA_HOST=0.0.0.0:8080
ollama serve &

echo "Waiting for Ollama to start..."
until curl -s http://localhost:8080/api/tags > /dev/null 2>&1; do
    sleep 2
done

echo "Pulling deepseek-r1:14b model..."
ollama pull deepseek-r1:14b

echo "Ollama is ready. Starting main application..."

exec python main.py