from app import app, init_models

# Initialize models once when WSGI server boots up
init_models()

if __name__ == '__main__':
    app.run()
