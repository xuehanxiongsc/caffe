import getopt,sys
import caffe
import score
   
def usage():
    print 'compute_metrics.py --model_file <caffemodel> ' \
          '--solver_file <solver file> --output_layer <output name>'

def main(argv):
    model_file = ''
    solver_file = ''
    output_layer = ''
    num_samples = 0
    try:
        opts, args = getopt.getopt(
            argv,
            "h:s:m:o:n:",
            ["help","solver_file=","model_file=","output_layer=","num_samples="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","--help"):
            usage()
            sys.exit()
        elif opt in ("-m","--model_file"):
            model_file = arg
        elif opt in ("-s","--solver_file"):
            solver_file = arg
        elif opt in ("-o","--output_layer"):
            output_layer = arg
        elif opt in ("-n","--num_samples"):
            num_samples = int(arg)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    solver.net.copy_from(model_file)
    names = [('%05d'%i) for i in xrange(num_samples)]
    score.seg_tests(solver, False, names, gt='Data2',layer=output_layer)
     
if __name__ == "__main__":
    main(sys.argv[1:])
