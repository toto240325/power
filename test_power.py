"""
Unit tests module for this project
Just run like this :
1. to test with the local utils :
export PYTHONPATH=/home/toto/utils ; cd ~/power ; venv ; python test_power.py
2. to test wiht the production utils :
cd ~/power ; venv ; python test_power.py
"""

import unittest
import power

class Testpower(unittest.TestCase):
    def test_values(self):
        #self.assertRaises(ValueError,friend, "wrong value for array ???"):
        pass
            
    def Test_assert_equals(self,p1,p2):
        self.assertEqual(p1,p2)
        
    # def test_power_ps4_OK(self):        
    #     # self.Test_assert_equals(power.main(),2)
    #     self.Test_assert_equals(power.check_ps4("192.168.0.40"),True)

    # def test_power_frigo_OK(self):        
    #     # self.Test_assert_equals(power.main(),2)
    #     self.Test_assert_equals(power.check_frigo(60*3),True)

    # def test_power_frigo_NOK(self):        
    #     # self.Test_assert_equals(power.main(),2)
    #     self.Test_assert_equals(power.check_frigo(1),False)
        
    # def test_power_full(self):        
    #     self.Test_assert_equals(power.main(),None)
        

if __name__ == "__main__":
    unittest.main()
