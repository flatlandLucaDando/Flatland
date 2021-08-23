import numpy as np

# wheight and height of the grid
width = 40
height = 30 

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
railway_example_1 = [[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],                          #1
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (1, 90),(2, 270), (10, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (0, 0), (0, 0), (0, 0)],    #2
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (8, 0), (1, 90), (1, 90),(2, 90), (10, 270),  (1, 90), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (1, 0), (0, 0), (0, 0), (0, 0)],      #3
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #4
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #5   
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #6
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),  (0,0),    (0,0),   (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),    (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #7
		 			[(0, 0) , (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 270), (10, 90), (1, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90),(8, 180), (1, 0),(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                   #8
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (8, 0),  (1, 90), (1, 90), (1, 90), (1, 90),  (1, 90),(1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 90), (10, 270), (10, 90), (2, 270), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90),(8, 180), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                #9
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                        #10
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                       #11
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #12
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],   #13
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 180), (2, 0), (0, 0), (0, 0), (0, 0)],   #14
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 0), (10, 180), (0, 0), (0, 0), (0, 0)],    #15
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #16
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 0),(10, 180), (0, 0), (0, 0), (0, 0)],  #17
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),(1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (10, 0),(10, 180), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 180), (2, 0), (0, 0), (0, 0), (0, 0)],  #18
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (1, 0), (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (2, 180), (2, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], #19
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],    #20
		 			[(0, 0),  (8, 0), (1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(2, 90),(5, 90),(2, 270), (10, 90), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 270), (10, 90), (2, 90),(10, 270), (10, 90), (2, 270), (1, 90), (1, 90),(1, 90),(1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (1, 0), (0, 0), (0, 0),(0, 0)],    #21
		 			[(7, 270),(2, 90),(1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(2, 90),(2, 90), (10, 270), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 90), (10, 270), (1, 90), (1, 90), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90),  (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (0, 0), (0, 0),(0, 0)],    #22
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #23
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #24
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #25
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #26
		 			[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]   #27

# Station positions are defined by the position x,y on the grid
station_a = (21, 1)
station_b = (21, 37)
station_c = (8, 14)
station_d = (1, 36)
station_e = (15, 49)

# An array more compact to store the different stations, for each station the array conteins the position (x,y) of the station, the maximum
# velocity supported by the line (high velocity, regional...)
stations = [[station_a, 1.], [station_b, 1.], [station_c, 0.5], [station_d, 0.5], [station_e, 1.]]

# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
# Each row represent a different train

timetable_example_1 = [[(station_a, station_b, station_e), (10 ,50, 80), (1.)],   		 # agent 0   high velocity
			 [(station_b, station_a),(0, 38), (1. / 2.)],                     			 # agent 1   Intercity
             [(station_c, station_b), (30, 109), (1.)],                  	             # agent 2   High velocity
             [(station_d, station_e), (30, 98), (1. / 4.)],                             # agent 3   Regional
             [(station_e, station_d), (50, 107), (1. / 4.)],					         # agent 4   Regional
             [(station_d, station_c), (60, 120), (1. / 2.)]]                              # agent 5   Intercity


# Different velocities in different lines.
speed_ration_map_lines = [1.       ,  # High velocity lines
						  1. / 2.  ]  # Regional lines

# Lines are represented by a rectangle between two stations
# The presence of an agent in the line is checked by checking if it is in the rectangle or not
line_a_b = [[20, 0],
			[21, 36]] # high velocity

line_b_e = [[15, 36],
			[21, 51]] # high velocity

line_a_c = [[8, 0],
			[21, 12]] # regional

line_c_d = [[1, 12],
			[8, 35]] # regional

line_d_e = [[1, 35],
			[15, 51]] # regional

