pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/neil-ravat/mlops-iris-demo.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Retrain Model') {
            steps {
                sh 'python3 retrain.py'
            }
        }

        stage('Deploy Model') {
            steps {
                sh '''
                # Copy new model to Flask app directory
                cp iris_model.pkl /home/ubuntu/app/

                # Restart Flask API (if using nohup)
                pkill -f app.py || true
                nohup python3 /home/ubuntu/app/app.py >/tmp/ml_api.out 2>&1 &
                '''
            }
        }
    }
}
