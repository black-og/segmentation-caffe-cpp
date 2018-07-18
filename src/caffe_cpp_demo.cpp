#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>

using namespace caffe;
using namespace std;
using namespace cv;

class Segmenter {
    public:
        Segmenter(const string& modelFile, const string& trainedFile);
        void Seg_Inference(const cv::Mat& img);

    private:
        void WrapInputLayer(vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, vector<cv::Mat>* input_channels);

    private:
        boost::shared_ptr<Net<float> > net_;
        cv::Size input_geometry;
        int num_channels;
};


int scanFiles(vector<string> &fileList, string inputDirectory)
{
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char* str = inputDirectory.c_str();

    p_dir = opendir(str);   
    if( p_dir == NULL)
    {
        cout<< "can't open :" << inputDirectory << endl;
    }

    struct dirent *p_dirent;

    while ( p_dirent = readdir(p_dir))
    {
        string tmpFileName = p_dirent->d_name;
        if( tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else
        {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList.size();
}

int Argmax(const vector<float> tmp){
    if(tmp.empty())
        return 0;
    int idx = 0;
    float max_elem = tmp[0];
    for(int i = 0; i < tmp.size(); i++){
        if(tmp[i] > max_elem){
            idx = i;
            max_elem = tmp[i];
        }
    }
    return idx;
}



Segmenter::Segmenter(const string& modelFile, const string& trainedFile) {
    Caffe::set_mode(Caffe::GPU);
    net_.reset(new Net<float>(modelFile, TEST));
    net_->CopyTrainedLayersFrom(trainedFile);

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels = input_layer->channels();
    input_geometry = cv::Size(input_layer->width(), input_layer->height());

}

void Segmenter::WrapInputLayer(vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_geometry.width;
    int height = input_geometry.height;
    
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < num_channels; i++){
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
    
}

void Segmenter::Preprocess(const cv::Mat& img, vector<cv::Mat>* input_channels) {
    cv::Mat sample;
    if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    
    cv::Mat sample_resized;
    if (sample.size() != input_geometry)
        cv::resize(sample, sample_resized, input_geometry);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else 
        sample_resized.convertTo(sample_float, CV_32FC1);
    
    sample_float = sample_float - cv::Scalar(100, 100, 100);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, *input_channels);    
}

void Segmenter::Seg_Inference(const cv::Mat& img) {

    vector<cv::Mat> input_channels;
    int featuremap_size = input_geometry.width * input_geometry.height;
    WrapInputLayer(&input_channels);
    
    Preprocess(img, &input_channels);
    
    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + featuremap_size * output_layer->channels();
    vector<float> output(begin, end);
    unsigned char result_map[featuremap_size];
    vector<unsigned char> tt;
    for(int i = 0; i < featuremap_size; i++){
        vector<float> tmp;
        for(int j = 0; j < output_layer->channels(); j++){
            tmp.push_back(output[i + j * featuremap_size]);
        }
        result_map[i] = 20 * Argmax(tmp);
    }
    cv::Mat result_img(input_geometry.height, input_geometry.width, CV_8UC1, result_map);
    imshow("segment_image",result_img);
    waitKey(0);
}

int main() {
    string modelFile = "../models/deploy.prototxt";
    string trainedFile = "../models/segment.caffemodel";
    Segmenter segmenter(modelFile, trainedFile);

    string image_path = "../data";
    vector<string> filename;
    int numImg = scanFiles(filename, image_path);
    if(numImg == 0){
        cout << "The directory is empty！" << endl;
        return -1;
    }
    for(vector<string>::iterator it = filename.begin(); it != filename.end(); it++){
        string image_name = image_path + "/" + (*it);
        cv::Mat img = cv::imread(image_name);
        if(img.empty() || !img.data){
			cout << "Fail to read image!" << endl;
			return -1;
		}
    segmenter.Seg_Inference(img);
    
    }
    return 0;
}