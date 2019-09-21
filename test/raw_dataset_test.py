import unittest
import raw_dataset.kitti_2012 as kitti_2012
import raw_dataset.kitti_2015 as kitti_2015
import raw_dataset.freiburg_driving as freiburg_driving
import raw_dataset.freiburg_monkaa as freiburg_monkaa
import raw_dataset.freiburg_flying as freiburg_flying
import timeit


class MyTestCase(unittest.TestCase):
    def test_freiburg_and_kitti_raw_datasets(self):
        a = timeit.default_timer()

        # create instances
        flying = freiburg_flying.split_dataset([1, 1, 1])
        monkaa = freiburg_monkaa.split_dataset([1, 1, 0])
        driving = freiburg_driving.split_dataset([1, 1, 0])
        kitti2012 = kitti_2012.split_dataset([1, 1, 1])
        kitti2015 = kitti_2015.split_dataset([1, 1, 1])

        self.assertEqual(len(flying.full_dataset['training_set']), 22390)
        self.assertEqual(len(flying.full_dataset['test_set']), 4370)

        self.assertEqual(len(monkaa.full_dataset['training_set']), 8664)
        self.assertEqual(len(monkaa.full_dataset['test_set']), 0)

        self.assertEqual(len(driving.full_dataset['training_set']), 4400)
        self.assertEqual(len(driving.full_dataset['test_set']), 0)

        self.assertEqual(len(kitti2012.full_dataset['training_set']), 194)
        self.assertEqual(len(kitti2012.full_dataset['test_set']), 195)

        self.assertEqual(len(kitti2015.full_dataset['training_set']), 200)
        self.assertEqual(len(kitti2015.full_dataset['test_set']), 200)
        print("Elapsed time: %.4f" % (timeit.default_timer() - a))



if __name__ == '__main__':
    unittest.main()
