def index():
    if request.method == "GET":
        return render_template("index.html")