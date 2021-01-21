from flask import Flask, render_template,  request
from io import BytesIO
import base64
import Image_classify
# from flask_script import Manager

app = Flask(__name__)
# manager = Manager(app)
image_recognise = Image_classify.Image_classify()


@app.route('/jquery-3.4.1.min.js')
def return_src():
    return render_template('jquery-3.4.1.min.js')


@app.route('/', methods=['GET', 'POST'])
def upload():
    imageValue = '-'
    if request.method == 'POST':
        if 'image' in request.form:
            image_data = request.form.get('image')
            # 图片转化为base64的url格式 如：data:image/png;base64,iVBORw0KGg 需要把data:image/png;base64,去掉
            image_data = image_data[22:]
            # decode image from base64 formate to utf-8
            image_data = base64.b64decode(image_data)
            # decode image from utf-8 to binary
            image_data = BytesIO(image_data)
            imageValue = image_recognise.recognise(image_data)
            # imageValue = 'recognize image '

            # image = Image.open(image_data)
            # image.show()
            print(f"image receive value {imageValue} ")
            return str(imageValue)
        else:
            print('uploading image have got into post method but image is not in the request.form ')
    else:
        print('uploading get menthod')
    print("return pages imageValue " + imageValue)
    return render_template("draw_bord.html")


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run()
    # manager.run()
