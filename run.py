"""
Ponto de entrada da aplicação.

Uso:
    python run.py

A aplicação sobe em http://localhost:5000 por padrão. Defina a variável de
ambiente PORT para usar outra porta.
"""

from app.server import app

if __name__ == "__main__":
    import os
    porta = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=porta, debug=False)
