/**
 *    author: ps_19
**/
#include <bits/stdc++.h>
using namespace std;
#define fastio() ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0)
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#define ll  long long
#define ld  long double
#define endl "\n"
#define pb  push_back
#define fill(a,val) memset(a,val,sizeof(a))
#define ff  first
#define ss  second
#define test  ll t; cin>>t; while(t--)
#define loop(i,a,b)  for(ll i=a;i<b;i++)
#define loopr(i,a,b) for(ll i=a;i>=b;i--)
#define pii pair<ll,ll>
#define all(v) v.begin(),v.end()
const ll mod     = 1000*1000*1000+7;
const ll inf     = 1ll*1000*1000*1000*1000*1000*1000 + 7;
const ll mod2    = 998244353;
const ll N       = 1000 + 10;
const ld pi      = 3.141592653589793;
ll power(ll x,ll y,ll p = LLONG_MAX ){ll res=1;x%=p;while(y>0){if(y&1)res=(res*x)%p;y=y>>1;x=(x*x)%p;}return res;}
ll ncr(ll n,ll r,ll m){if(r>n)return 0;ll a=1,b=1,i;for(i=0;i<r;i++){a=(a*n)%m;--n;}while(r){b=(b*r)%m;--r;}return (a*power(b,m-2,m))%m;}


                      // topology[layers] vector contains the number of layers in that particular layer.
                     // firstly we will assume that all layers of neuron are tightly connected to each other and their is only one layer which have any constant value called bias neuron.
using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology){
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename){
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals){
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals){
    targetOutputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

struct Connection{
 double weight;
 double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

// *************** class Neurons **************

class Neuron               // Neuron class objects should contain minimum information like as given
{
 public:
     Neuron(unsigned numOutputs,unsigned myIndex);       // numOutput contains numbers of layer in the next layer.

     void setOutputVal(double val){
        m_outputVal=val;
     }
     double getOutputVal(void) const{
        return m_outputVal;
     }
     void feedForward(const Layer &prevLayer);
     void calcOutputGradients(double targetVal);
     void calcHiddenGradients(const Layer &nextLayer);
     void updateInputWeights(Layer &prevLayer);

 private:
     static double eta; // [0.0..1.0] overall net training rate
     static double alpha;  // [0.0..n] multiplier of last weight change (momentum)
     static double transferFunction(double x);
     static double transferFunctionDerivative(double x);
     static double randomWeight(void){
           return rand()/double(RAND_MAX);      // It will give random value between 0 and 1 defined in library cstdlib
     }
     double sumDOW(const Layer &nextLayer) const;
     double m_outputVal;
     vector<Connection> m_outputWeights;

     unsigned m_myIndex;
     double m_gradient;
};


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer){
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
                                              // the work of transfer function is only to scale our output and set the value in particular range here tanh[x] function has been used
return tanh(x);

}

double Neuron::transferFunctionDerivative(double x){

return (1-x*x);

}

void Neuron::feedForward(const Layer &prevLayer){
     double sum=0.0;   //contains the sum of output of previous neurons
                     //which is input for present neuron
     for(unsigned n=0;n<prevLayer.size();++n){
        sum+=prevLayer[n].getOutputVal()*prevLayer[n].m_outputWeights[m_myIndex].weight;   // here prevLayer[n].m_outputVal will also work

     }
      m_outputVal=Neuron::transferFunction(sum);


}



     // Neurons properties has been defined
Neuron::Neuron(unsigned numOutputs,unsigned myIndex){
  for(unsigned c=0 ; c < numOutputs ; ++c){            // c for connections

    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight=randomWeight();

  }

  m_myIndex=myIndex;

}

// ************* class net *************  in downward

class Net{

public:
  Net(const vector<unsigned> &topology);  //const means a constant value
  void feedForward(const vector<double> &inputVals);      //the ""("&")inputVals"" ampersant sign shows that it will take values from inputVals by reference here not whole array has been passed.
  void backProp(const vector<double> &targetVals);
  void getResults(vector<double> &resultVals) const;
  double getRecentAverageError(void) const { return m_recentAverageError; }


private:
   vector<Layer> m_layers;  // creating layers of neuron is better idea than to create a 2-D array
                            // firstly a vector of layers data type has been created
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net :: backProp(const vector<double> &targetVals){
  // calculate overall error
  // here root means square error
  Layer &outputLayer=m_layers.back();
  m_error=0.0;

  for(unsigned n=0;n<outputLayer.size()-1;++n){
    double delta=targetVals[n]-outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size()-1;  //average square error
  m_error = sqrt(m_error);     //RMS error

  // Implementation of recent average measurement

  m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

  //calculate output layer gradients

  for(unsigned n=0;n<outputLayer.size()-1;++n){
     outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  //calculate gradient on hidden layers

  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

  //for all layers from outputs to hidden layer.

  //update connection weights
  for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
  }
}

void Net::feedForward(const vector<double> &inputVals){
   assert(inputVals.size()==m_layers[0].size()-1);      // error handling checking
   for(unsigned i=0 ; i<inputVals.size() ; ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);           // setOutputVals is function that formulates the inputVals and gives required result
   }
   // forwarding message from one neuron to another neuron the input of one neuron is the output of another neuron

   for(unsigned layerNum=1;layerNum<m_layers.size();++layerNum){
        Layer &prevLayer=m_layers[layerNum-1];          // pointer that points to previous neuron of its type

        for(unsigned n=0;n<m_layers[layerNum].size()-1;++n){
            m_layers[layerNum][n].feedForward(prevLayer);                       //feedForward is neuron member function which perform actions on input and forwards output
                                                         // layerNum is number of layer and n is nth neuron.
        }


   }
}


Net::Net(const vector<unsigned> &topology){
   unsigned numLayers=topology.size();
   for(unsigned layerNum=0;layerNum<numLayers;++layerNum){
          m_layers.push_back(Layer());   // a new layer of neuron has been created using a=one vector.
                                        // now to fill input anther loop is required.
         unsigned numOutputs= layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
         if( layerNum == (topology.size()-1)){
            numOutputs=0;
         }
         else{
            numOutputs=topology[layerNum+1];
         }

          //We have a new layer, now fill it with neurons and its information into it
         //add a bias neuron( a neuron with constant value(as assumed in this code)) in each layer.
       for(unsigned neuronNum=0;neuronNum<=topology[layerNum]; ++neuronNum){
           m_layers.back().push_back(Neuron(numOutputs,neuronNum));      // passing neuronNum for denoting its index value in m_layer
           cout << "Made a Neuron" << endl;
       }                                   // topology[layerNum] is total numbers of neuron in that layer
           m_layers.back().back().setOutputVal(1.0);                                 // less than or equal to topo[layerNum] because it is assumed that every layer have one bias layer- a layer with constant value adding to input to give output.
   }

}

void showVectorVals(string label, vector<double> &v){
    cout<<label<<" ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout<<v[i]<<" ";
    }
    cout<<endl;
}

int main(){
TrainingData trainData("/tmp/trainingData.txt");
vector<unsigned> topology;
Net myNet(topology);   // topology is a constructor of Net class.


  vector<double> inputVals;
  vector<double> targetVals;
  vector<double> resultVals;
  int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "<< myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;


}
