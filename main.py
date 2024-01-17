import argparse
import trainer
import plot


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--', type=int, default=0)


def main():
    parser = argparse.ArgumentParser()

    # Choose which test to run. None are run if left blank
    parser.add_argument('--test', type=str, required=False)

    # Choose which plot to generate. None are generated if left blank
    parser.add_argument('--plot', type=str, required=False)
    args = parser.parse_args()

    #Run Test, Store Data
    if args.test == 'assoc_comp':
        trainer.modCompareAA()
    elif args.test == 'OnCont-L1':
        trainer.run_onCont_L1()
    elif args.test == 'OnCont-L3':
        trainer.run_onCont_L3()
    elif args.test == 'nsEncode-L1':
        trainer.nsEncode_L1()
    elif args.test == 'nsEncode-L3':
        trainer.nsEncode_L3()
    elif args.test == 'recog':
        trainer.recog()
    elif args.test == 'arch_compare':
        trainer.run_arch_compare()
    elif args.test is not None:
        assert False, 'Invalid test argument. Argument must be from list [assoc_comp, OnCont-L1, OnCont-L3, nsEncode-L1, ' \
                      'nsEncode-L3, recog, arch_compare]'


    # Select and Generate Plots
    if args.plot == 'OnCont-L1':
        plot.plot_sensit()
        plot.plot_cont()
        plot.plot_cont_cumul()
    elif args.plot == 'OnCont-L3':
        plot.plot_sensit_tree()
        plot.plot_cont_tree()
        plot.plot_cont_cumul_tree()
    elif args.plot == 'nsEncode-L1':
        plot.plot_noisy_online()
    elif args.plot == 'nsEncode-L3':
        plot.plot_noisyTree_online()
    elif args.plot == 'recog':
        plot.plot_recog_all()
    elif args.plot == 'arch_compare':
        plot.plot_tree_aa()
        plot.plot_accTest_tree()
        plot.plot_recognition_tree()
    elif args.plot is not None:
        assert False, 'Invalid test argument. Argument must be from list [assoc_comp, OnCont-L1, OnCont-L3, nsEncode-L1, ' \
                      'nsEncode-L3, recog, arch_compare]'


    if args.plot is None and args.test is None:
        assert False, 'No Arguments Inputted. Must input a test and/or a plot argument'

    return




if __name__ == "__main__":
    main()