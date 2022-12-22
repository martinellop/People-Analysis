import torch


def L1_distance(x:torch.tensor, y: torch.tensor):
    return torch.sum(torch.abs(x-y), -1)

def L2_distance(x:torch.tensor, y: torch.tensor):
    return torch.sqrt(torch.sum(torch.pow(x-y,2), -1))


class PeopleDB:
    def __init__(self, dist_function, dist_threshold:float, frame_memory:int, device:torch.device=None):
        self._dist_function_ = dist_function
        self._dist_threshold_ = dist_threshold
        self._frame_memory_ = frame_memory

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self._current_frame_ = 0
        self._last_id_generated = 0

    def Get_ID(self, descriptor:torch.tensor, update_record:bool=True):
        '''
        Retrieve the ID of a descriptor similar to the given one, or create a new ID if there is no similar vector.
        In case of match, the vector stored in database can also be updated with the new one.
        Returns the ID and a boolean pointing if the ID has been created for this instance (rather than reusing an existing one).
        '''
        if self._last_id_generated == 0:
            self._vectors_ = torch.zeros((0,descriptor.shape[0]), dtype=torch.float64, device=self.device)  # descriptors
            self._ids_ = torch.zeros((0,), dtype=torch.long, device=self.device)                           # IDs associated to corresp. descriptor
            self._last_update_ = torch.zeros((0,), dtype=torch.long, device=self.device)                   # frame number of last time the descriptor has been used/updated.
            return self._Create_new_record_(descriptor), True

        #print("self._vectors_shape: ", self._vectors_.shape)
        distances = self._dist_function_(self._vectors_, descriptor)
        #print("distances_shape: ", distances.shape)
        idx = torch.argmin(distances)
        #print("idx_shape: ", idx.shape)

        #print(f"Min dist: {distances[idx]} in {distances}")

        if distances[idx] <= self._dist_threshold_:
            #we got a match
            if update_record:
                self._vectors_[idx] = descriptor
            self._last_update_[idx] = self._current_frame_
            return self._ids_[idx], False
        else:
            #this seems to be a new target
            return self._Create_new_record_(descriptor), True

    def _Create_new_record_(self, new_vector:torch.tensor):
        self._last_id_generated += 1
        self._vectors_ = torch.cat((self._vectors_, new_vector.reshape((1,-1))))
        new_id = torch.tensor((self._last_id_generated), dtype=torch.long,device=self.device).reshape(1,1)
        self._ids_ = torch.cat((self._ids_, new_id))
        last_update = torch.tensor((self._current_frame_), dtype=torch.long,device=self.device).reshape(1,1)
        self._last_update_ = torch.cat((self._last_update_, last_update))
        return self._last_id_generated

    def Update_Frame(self):
        self._current_frame_ += 1

        # let's remove vectors which are too old
        if len(self._last_update_) == 0:
            return

        #print(self._vectors_.shape, self._last_update_.shape, self._ids_.shape)
        toKeep =  self._last_update_ > (self._current_frame_ - self._frame_memory_)
        toKeep = toKeep.flatten()
        self._vectors_ = self._vectors_[toKeep]
        self._last_update_ = self._last_update_[toKeep]
        self._ids_ = self._ids_[toKeep]

        #print(self._vectors_.shape, self._last_update_.shape, self._ids_.shape)

    def Clear(self):
        #necessary?

        print("\n\nRemaining IDs in DB: ")
        print(self._ids_, "\n\n")

        del self._vectors_
        del self._ids_
        del self._last_update_
