# coding=utf-8
from flask import Flask, render_template,request,redirect,flash
from flask_bootstrap import Bootstrap
from flask_cors import *
import pandas as pd
app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = '123456'
bootstrap = Bootstrap(app)
CORS(app, supports_credentials=True)
from flask import Flask, render_template, request, jsonify

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('img/favicon.ico')

@app.route('/')
def home():
    return redirect("/index")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/vertical')
def vertical():
    return render_template('vertical.html')

@app.route('/horizontal')
def horizontal():
    return render_template('horizontal.html')

@app.route('/transfer')
def transfer():
    return render_template('transfer.html')

@app.route('/horizontal_gmm')
def horizontal_gmm():
    import GMM_horizontal
    start_feature = request.args.get("start_feature4", '')
    end_feature = request.args.get("end_feature4", '')
    final_global_means = ''
    scores = ''
    device1_means = ''
    device1_score = ''
    device2_means = ''
    device2_score = ''
    final_global_covariances = ''
    device1_covariances = ''
    device2_covariances = ''
    if start_feature != '' and end_feature != '':
        start_feature = int(start_feature)
        end_feature = int(end_feature)
        final_global_means, final_global_covariances, scores, device1_means, device1_covariances, device1_score, device2_means, device2_covariances, device2_score = GMM_horizontal.GMM_horizontal(start_feature,end_feature)
    return render_template('horizontal_gmm.html', final_global_means=final_global_means, scores=scores,
                           device1_means=device1_means, device1_score=device1_score,
                           device2_means=device2_means, device2_score=device2_score,
                           final_global_covariances=final_global_covariances,device1_covariances=device1_covariances,device2_covariances=device2_covariances)

@app.route('/horizontal_kmeans')
def horizontal_kmeans():
    import kmeans_horizontal
    start_feature = request.args.get("start_feature",'')
    end_feature = request.args.get("end_feature",'')
    final_global_centroids = ''
    scores = ''
    device1_centroids = ''
    device1_score = ''
    device2_centroids = ''
    device2_score = ''
    if start_feature != '' and end_feature != '':
        start_feature = int(start_feature)
        end_feature = int(end_feature)
        final_global_centroids, scores, device1_centroids, device1_score, device2_centroids, device2_score = kmeans_horizontal.kmeans_horizontal(start_feature,end_feature)
    return render_template('horizontal_kmeans.html', final_global_centroids=final_global_centroids, scores=scores,device1_centroids=device1_centroids,device1_score=device1_score,device2_centroids=device2_centroids,device2_score=device2_score)

@app.route('/vertical_gmm')
def vertical_gmm():
    import GMM_vertical
    start_feature = request.args.get("start_feature5", '')
    end_feature = request.args.get("end_feature5", '')
    final_global_means = ''
    scores = ''
    device1_means = ''
    device1_score = ''
    device2_means = ''
    device2_score = ''
    final_global_covariances = ''
    device1_covariances = ''
    device2_covariances = ''
    if start_feature != '' and end_feature != '':
        start_feature = int(start_feature)
        end_feature = int(end_feature)
        final_global_means, final_global_covariances, scores, device1_means, device1_covariances, device1_score, device2_means, device2_covariances, device2_score = GMM_vertical.GMM_vertical(start_feature,end_feature)
    return render_template('vertical_gmm.html', final_global_means=final_global_means, scores=scores,
                           device1_means=device1_means, device1_score=device1_score,
                           device2_means=device2_means, device2_score=device2_score,
                           final_global_covariances=final_global_covariances,device1_covariances=device1_covariances,device2_covariances=device2_covariances)

@app.route('/vertical_kmeans')
def vertical_kmeans():
    import kmeans_vertical
    start_feature1 = request.args.get("start_feature1",'')
    end_feature2 = request.args.get("end_feature1",'')
    final_global_centroids = ''
    scores = ''
    device1_centroids = ''
    device1_score = ''
    device2_centroids = ''
    device2_score = ''
    if start_feature1 != '' and end_feature2 != '':
        start_feature1 = int(start_feature1)
        end_feature2 = int(end_feature2)
        final_global_centroids, scores, device1_centroids, device1_score, device2_centroids, device2_score = kmeans_vertical.kmeans_vertical(start_feature1,end_feature2)
    return render_template('vertical_kmeans.html', final_global_centroids=final_global_centroids, scores=scores,device1_centroids=device1_centroids,device1_score=device1_score,device2_centroids=device2_centroids,device2_score=device2_score)

@app.route('/transfer_gmm')
def transfer_gmm():
    import GMM_transfer_learning
    start_feature = request.args.get("start_feature3", '')
    end_feature = request.args.get("end_feature3", '')
    final_global_means = ''
    scores = ''
    device1_means = ''
    device1_score = ''
    device2_means = ''
    device2_score = ''
    final_global_covariances = ''
    device1_covariances = ''
    device2_covariances = ''
    if start_feature != '' and end_feature != '':
        start_feature = int(start_feature)
        end_feature = int(end_feature)
        final_global_means, final_global_covariances, scores, device1_means, device1_covariances, device1_score, device2_means, device2_covariances, device2_score = GMM_transfer_learning.GMM_transfer_learning(start_feature,end_feature)
    return render_template('transfer_gmm.html', final_global_means=final_global_means, scores=scores,
                           device1_means=device1_means, device1_score=device1_score,
                           device2_means=device2_means, device2_score=device2_score,
                           final_global_covariances=final_global_covariances,device1_covariances=device1_covariances,device2_covariances=device2_covariances)


@app.route('/transfer_kmeans')
def transfer_kmeans():
    import kmeans_transfer_learning
    start_feature = request.args.get("start_feature2",'')
    end_feature = request.args.get("end_feature2",'')
    final_global_centroids = ''
    scores = ''
    device1_centroids = ''
    device1_score = ''
    device2_centroids = ''
    device2_score = ''
    if start_feature != '' and end_feature != '':
        start_feature = int(start_feature)
        end_feature = int(end_feature)
        final_global_centroids, scores, device1_centroids, device1_score, device2_centroids, device2_score = kmeans_transfer_learning.kmeans_transfer_learning(start_feature,end_feature)
    return render_template('transfer_kmeans.html', final_global_centroids=final_global_centroids, scores=scores,device1_centroids=device1_centroids,device1_score=device1_score,device2_centroids=device2_centroids,device2_score=device2_score)


if __name__ == '__main__':
    app.run(debug=True,port="5000",host="0.0.0.0")
