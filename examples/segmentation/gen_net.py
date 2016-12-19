import getopt,sys
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from fcn16 import fcn16
from fcn16_1 import fcn16_1
from fcn32 import fcn32
from fcn16_enet import fcn16_enet
   
net_map = {
    'fcn16': fcn16,
    'fcn32': fcn32,
    'fcn16_1': fcn16_1,
    'fcn16_enet': fcn16_enet
}    
    
def usage():
    print 'gen_net.py --train_file <train data> ' \
          '--val_file <valiation data> --model_name <model name>'\
          '--batch_size <batch size>'

def main(argv):
    train_file = ''
    val_file = ''
    model_name = 'fcn16'
    batch_size = 64
    try:
        opts, args = getopt.getopt(
            argv,
            "h:t:v:b:m:",
            ["train_file=","val_file=","model_name=","batch_size="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","--help"):
            usage()
            sys.exit()
        elif opt in ("-t","--train_file"):
            train_file = arg
        elif opt in ("-v","--val_file"):
            val_file = arg
        elif opt in ("-m","--model_name"):
            model_name = arg
        elif opt in ("-b","--batch_size"):
            batch_size = int(arg)

    with open(('%s.prototxt' % model_name), 'w') as f:
        f.write(str(net_map[model_name](train_file, val_file, batch_size)))
    
if __name__ == "__main__":
    main(sys.argv[1:])
