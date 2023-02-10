import torch
import random

class PeopleDB:
    def __init__(self, dist_function, dist_threshold:float, frame_memory:int, max_descr_per_id:int, positions_buffer:int=30, device:torch.device=None):
        self._dist_function_ = dist_function
        self._dist_threshold_ = dist_threshold
        self._frame_memory_ = frame_memory
        self._max_descr_per_id_ = max_descr_per_id  # how many descriptors to store for each identity?

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self._current_frame_ = 0
        self._last_id_generated = 0

        self._max_positions = positions_buffer  #how many positions to save? -1 to disable this feature.
        if(positions_buffer > 0):
            self._last_positions = {}   #dictionary in which, for each identity, will be stored the last n positions

    def Get_ID(self, descriptor:torch.Tensor):
        '''
        Retrieve the ID of a descriptor similar to the given one, or create a new ID if there is no similar vector.
        In case of match, the vector stored in database can also be updated with the new one.
        Returns the ID and a boolean pointing if the ID has been created for this instance (rather than reusing an existing one).
        '''
        if self._last_id_generated == 0:
            self._vectors_ = torch.zeros((0, self._max_descr_per_id_, descriptor.shape[0]), dtype=torch.float64, device=self.device, requires_grad=False)  # descriptors
            self._ids_ = torch.zeros((0,), dtype=torch.long, device=self.device, requires_grad=False)                           # IDs associated to corresp. descriptor
            self._counts_ = torch.zeros((0,), dtype=torch.int, device=self.device, requires_grad=False)                         # how many descriptors are stored for each id?
            self._last_update_ = torch.zeros((0,), dtype=torch.long, device=self.device, requires_grad=False)                   # frame number of last time the descriptor has been used/updated.
            return self._Create_new_record_(descriptor), True

        n_identities = self._ids_.shape[0]
        mean_distances = torch.zeros((n_identities), dtype=torch.float64, device=self.device, requires_grad=False)
        for i in range(n_identities):
            n_samples = self._counts_[i]
            id_distances = self._dist_function_(self._vectors_[i, :n_samples, :], descriptor.unsqueeze(0))
            mean_distances[i] = torch.sum(id_distances).flatten() / n_samples

        idx = torch.argmin(mean_distances)  # here we search for the identity which is closer to the probe.

        #print(f"mean_dists ({mean_distances.shape}): {mean_distances}. minimum at index {idx}")

        if mean_distances[idx] <= self._dist_threshold_:
            #we got a match

            if self._counts_[idx] < self._max_descr_per_id_:
                #there is still space for this identity
                self._vectors_[idx, self._counts_[idx].item()] = descriptor
                self._counts_[idx] += 1
            else:
                #we remove a random sample in order to insert this one.
                target_sample = random.randint(0, self._max_descr_per_id_-1)
                self._vectors_[idx, target_sample] = descriptor

            self._last_update_[idx] = self._current_frame_
            return self._ids_[idx], False
        else:
            #this seems to be a new target
            return self._Create_new_record_(descriptor), True

    def _Create_new_record_(self, new_vector:torch.Tensor):
        new_id = torch.zeros((1,self._max_descr_per_id_, new_vector.shape[0]), dtype=torch.float64, device=self.device, requires_grad=False)
        new_id[0,0] = new_vector
        #print("pre:",self._vectors_.shape)
        self._vectors_ = torch.cat((self._vectors_, new_id))
        #print("after:",self._vectors_.shape)


        self._last_id_generated += 1
        print("created ID ", self._last_id_generated)
        new_id = torch.tensor((self._last_id_generated), dtype=torch.long,device=self.device, requires_grad=False).reshape(1,1)
        self._ids_ = torch.cat((self._ids_, new_id))

        self._counts_ = torch.cat((self._counts_, torch.tensor((1), dtype=torch.int, device=self.device).reshape(1,1)))

        last_update = torch.tensor((self._current_frame_), dtype=torch.long,device=self.device, requires_grad=False).reshape(1,1)
        self._last_update_ = torch.cat((self._last_update_, last_update))
        return self._last_id_generated #returns the id

    def Update_Frame(self):
        self._current_frame_ += 1

        # let's remove vectors which are too old
        if len(self._last_update_) == 0:
            return

        #print(self._vectors_.shape, self._last_update_.shape, self._ids_.shape)
        toKeep =  self._last_update_ > (self._current_frame_ - self._frame_memory_)
        toKeep = toKeep.flatten()

        toRemove = ~toKeep
        ids_to_remove = self._ids_[toRemove]

        self._ids_ = self._ids_[toKeep]
        self._vectors_ = self._vectors_[toKeep]
        self._last_update_ = self._last_update_[toKeep]
        self._counts_ = self._counts_[toKeep]
        
        for id in ids_to_remove:
            print(f"Removed ID ", id.item())
            if self._max_positions > 0:
                del self._last_positions[int(id.item())]

        #print(self._vectors_.shape, self._last_update_.shape, self._ids_.shape)


    def Update_ID_position(self, personID:int, current_pos):

        if not personID in self._last_positions.keys():
            self._last_positions[personID] = []
        
        self._last_positions[personID].append(current_pos)
        if len(self._last_positions[personID]) > self._max_positions:
            self._last_positions[personID].pop(0)

        return self._last_positions[personID]
    
