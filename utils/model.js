let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
let isPredicting = false;
var IMAGENET_CLASSES = []
var currentName = null;
var currentLabel = 0;
var currentPicNum = 0;
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}

function enterName(){
  if(document.getElementById("input_name").value == ""){
    alert("Please Enter You Name !!");
  }else{
    stopPredicting();
    if(currentName != null){
      alert(`Please complete your traning. \n You have enter this Name ${currentName}`);
    }else{
      currentName = document.getElementById("input_name").value ;
      currentLabel = IMAGENET_CLASSES.length;
      IMAGENET_CLASSES.push({[currentLabel] : document.getElementById("input_name").value}) 
      document.getElementById("slang").innerText = "Getting you In !! Please take at least 10 pics.";
      document.getElementById("body_text").innerText = "";
    }
  }
}

function handleButton(){
  console.log(document.getElementById("input_name").value);
  if(currentName == null){
    alert("Enter your name before taking picture !!");
  }else{
    currentPicNum++;
    document.getElementById("pic_num").innerText = currentPicNum;
    label = currentLabel;
    console.log(IMAGENET_CLASSES);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);  
  }
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      var predictions = null;
      if(model == undefined){
        predictions = mobilenet.predict(img);
      }else{
        predictions = model.predict(activation) 
      }
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
		for(var i=0;i<IMAGENET_CLASSES.length;i++){
      if(IMAGENET_CLASSES[i][classId]!= undefined){
        if(isPredicting == true){
          document.getElementById("slang").innerText = "Got You :)"
          document.getElementById("body_text").innerText = `You are ${IMAGENET_CLASSES[i][classId]}`
        }
        break;
      }
    }
    if(isPredicting == true){
      if(model == undefined ){
        document.getElementById("slang").innerText = "Do I know you ? "
        document.getElementById("body_text").innerText = "What is your Name ?"
      }
    }
    predictedClass.dispose();
    await tf.nextFrame();
  }
}
function doTraining(){
  currentLabel = null;
  currentName = null;
  currentPicNum = 0;
  document.getElementById("pic_num").innerText = currentPicNum;
  document.getElementById("input_name").value = "";
  train();
  alert("Training Done!")
  startPredicting()
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
  startPredicting()
}


init();