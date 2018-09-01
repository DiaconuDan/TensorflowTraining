const parse = require('csv-parse/lib/sync') ;

const fs = require('fs') ; 
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node') ; 

const file = fs.readFileSync('testing_files/train_clean.csv') ;
const parsed = parse(file, { comment: '#'})

//  console.log(parsed);

const parsedSplit = parsed.reduce( (acc,el) => {
    let y = el[0] ;
    let x = el.slice(1) ;
    acc.x.push(x);
    acc.y.push(y);
    // console.log(acc);
    return acc ;
}, {x:[],y:[]})

// console.log('done!!!!!!!!!!s');
//  console.log(parsedSplit)


const xs = tf.tensor2d(parsedSplit.x, [parsedSplit.x.length, 7]) ; // tensor for features
const ys = tf.tensor2d(parsedSplit.y, [parsedSplit.y.length,1]) ;// output, labels

const model = tf.sequential()
model.add(tf.layers.dense({ units : 100, activation:'sigmoid', inputShape: [7]}));
// model.add(tf.layers.dropout({rate: 0.3})) ;s
model.add(tf.layers.dense({units:1, activation: 'sigmoid'}));


model.compile({ optimizer: 'rmsprop', loss: 'binaryCrossentropy'})


model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
        onEpochEnd: async function (epoch, log) {
            console.log('Epoch' +epoch +' : ' +' loss =  '+ log.loss);
        },
        onTrainEnd: async function() {
            const fileTest = fs.readFileSync('testing_files/test_clean.csv') ;
            const parsedTest = parse(fileTest, { comment:'#'}) ;

            const xsTest = tf.tensor2d(parsedTest, [parsedTest.length,7])

            const ysTest = model.predict(xsTest) ;

            ysTest.print() ;

            let ysTestArr = Array.from(ysTest.dataSync())
            console.log(ysTestArr.length, ysTestArr.reduce((acc,el) => {
                acc += el > 0.5 ? 1 : 0 
                return acc 
            }, 0 ))

            

        }
    }
})