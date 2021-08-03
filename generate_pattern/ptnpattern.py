import numpy as np

class combination:
    def __init__(self,dim=16):
        self.dim   = dim
        self.image = np.zeros((dim,dim))
    def _add_cross(self,value=1):
        #         #
        #        ###
        #         #
        x     = np.random.randint(1,15)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x-1,y],[x+1,y],[x,y-1],[x,y+1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_block(self,value=1):
        #         ###
        #         ###
        #         ###
        x     = np.random.randint(1,15)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x-1,y],[x+1,y],[x,y-1],[x,y+1],[x-1,y-1],[x-1,y+1],[x+1,y-1],[x+1,y+1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_triangle_up(self,value=1):
        #          #
        #         ###
        x     = np.random.randint(1,15)
        y     = np.random.randint(1,16)
        point = np.array([[x,y],[x-1,y],[x+1,y],[x,y-1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_triangle_left(self,value=1):
        #         #
        #        ##
        #         #
        x     = np.random.randint(1,16)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x-1,y],[x,y-1],[x,y+1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_triangle_right(self,value=1):
        #         #
        #         ##
        #         #
        x     = np.random.randint(0,15)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x+1,y],[x,y-1],[x,y+1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_triangle_down(self,value=1):
        #        ###
        #         #
        x     = np.random.randint(1,15)
        y     = np.random.randint(0,15)
        point = np.array([[x,y],[x-1,y],[x+1,y],[x,y+1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_U(self,value=1):
        #       # #
        #       ###
        x     = np.random.randint(1,15)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x-1,y+1],[x,y+1],[x-1,y+1],[x-1,y-1],[x,y-1],[x-1,y-1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def _add_H(self,value=1):
        #       # #
        #       ###
        #       # #
        x     = np.random.randint(1,15)
        y     = np.random.randint(1,15)
        point = np.array([[x,y],[x-1,y+1],[x-1,y],[x-1,y-1],[x+1,y+1],[x+1,y],[x+1,y-1]])
        point = tuple(point.T.tolist())
        if self.image[point].sum()>0:
            return False
        else:
            self.image[point]= value
        return True
    def add_pattern(self,a,v=1):
        if   a==0:
            return self._add_cross(v)
        elif a==1:
            return self._add_block(v)
        elif a==2:
            return self._add_triangle_up(v)
        elif a==3:
            return self._add_triangle_left(v)
        elif a==4:
            return self._add_triangle_right(v)
        elif a==5:
            return self._add_triangle_down(v)
        elif a==6:
            return self._add_U(v)
        elif a==7:
            return self._add_H(v)
    def get_no_cover_pattern(self,num=16):
        #choose pattern

        self.image = np.zeros((self.dim,self.dim))
        count = 0
        while count<num:
            a = np.random.randint(8)
            count+=self.add_pattern(a)
        return self.image
    def get_no_cover_pattern_demo(self,num=16):
        #choose pattern

        self.image = np.zeros((self.dim,self.dim))
        count = 0
        while count<num:
            a = count%8
            count+=self.add_pattern(a,a/7)
        return self.image
