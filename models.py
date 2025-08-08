import torch
import torch.nn as nn
import torch.nn.functional as F

# each reigon is representend with three values, mean, variance, and a constant values respectively. the reigon equation is exp(-(x-mean)^2 / 2*variance) * exp(c)
EULER_GAMMA = 0.57721566490153286060
SMALL_POSITIVE_FLOAT_FOR_STABILITY = 1e-23

class Word2VecCBOW(nn.Module):

    def __init__(self, vocab_size, embedding_size, num_negative_examples):
        super().__init__()

        # those two lines does not affect non-negative indices and they change the -1 to the last box index.
        self.num_vectors = vocab_size + 1
        self.num_negative_examples = num_negative_examples

        self.center_embeddings = nn.Embedding(self.num_vectors, embedding_size, padding_idx=self.num_vectors-1)
        self.context_embeddings = nn.Embedding(self.num_vectors, embedding_size, padding_idx=self.num_vectors-1)

        # Initialize the unity dimension
        self.center_embeddings.weight[self.num_vectors - 1].data.zero_()
        self.context_embeddings.weight[self.num_vectors - 1].data.zero_()

    def forward(self, x): # x is in the shape (batch_size, 1 + num_negative_samples + 2 * window_size) where each value is the index of a word

        x = (x + self.num_vectors) % self.num_vectors # changes -1 => self.num_vectors - 1, and does not affect the rest of the indices

        center_embedding = self.center_embeddings(x[:, :1]) # (batch_size, 1, embedding_size)
        negative_embedding = self.center_embeddings(x[:, 1: 1 + self.num_negative_examples]) # (batch_size, num_negative_examples, embedding_size)
        context_embedding = self.context_embeddings(x[:, 1 + self.num_negative_examples:])   # (batch_size, 2*window_size, embedding_size)

        context_sum = context_embedding.sum(dim=1, keepdim=True) # (batch_size, 1, embedding_size)

        positive_score = (center_embedding * context_sum).sum(dim=-1) 
        negative_score = (negative_embedding * context_sum).sum(dim=-1)       
        
        return positive_score, negative_score
    
    def get_similarity(self, word1, word2, condition_on_first_word = False): # word1 and word2 are indices
        # The parameter condition_on_first_word is not used and its here for general consistency with the implementaion of the other models (see word2box or word2ellipsoid)
        with torch.no_grad():
            # vector1, vector2 = self.center_embeddings(torch.tensor(word1)), self.center_embeddings(torch.tensor(word2))
            vector1, vector2 = self.context_embeddings(torch.tensor(word1)), self.context_embeddings(torch.tensor(word2))
            return F.cosine_similarity(vector1, vector2, dim=0).item()
        


class Word2BoxCBOW(nn.Module):

    # Every gumbel random variable has the same value for the beta (scale parameter) (which is equal to 1).
    def _initialize_boxes(self, num_boxes, embedding_size, min_value, max_value, dimension_width):
        with torch.no_grad():
            u_center_lower_bound = torch.zeros(num_boxes, embedding_size)
            u_context_lower_bound = torch.zeros(num_boxes, embedding_size)
            torch.nn.init.uniform_(u_center_lower_bound, min_value, max_value)
            torch.nn.init.uniform_(u_context_lower_bound, min_value, max_value)
            u_center_upper_bound = u_center_lower_bound + dimension_width
            u_context_upper_bound = u_context_lower_bound + dimension_width

            self.u_center_lower_bound.weight[:] = u_center_lower_bound
            self.u_center_upper_bound.weight[:] = u_center_upper_bound
            self.u_context_lower_bound.weight[:] = u_context_lower_bound
            self.u_context_upper_bound.weight[:] = u_context_upper_bound

            # Assign a unity box to the last index. The unity box lower bound is -inf and upper bound is inf, so that it does not affect the intersection operation
            self.u_context_lower_bound.weight[-1] = torch.full((1,embedding_size), -torch.inf)
            self.u_context_upper_bound.weight[-1] = torch.full((1,embedding_size), torch.inf)    
            self.u_center_lower_bound.weight[-1] = torch.full((1,embedding_size), -torch.inf)
            self.u_center_upper_bound.weight[-1] = torch.full((1,embedding_size), torch.inf) 


    def __init__(self, vocab_size, embedding_size, num_negative_examples):
        super().__init__()

        self.num_boxes = vocab_size + 1 # We add a unity box at the end.
        self.num_negative_examples = num_negative_examples

        self.u_center_lower_bound = nn.Embedding(self.num_boxes, embedding_size, padding_idx=self.num_boxes-1) # Location parameter value for the lower bound gumbel random variable (center embedding)
        self.u_center_upper_bound = nn.Embedding(self.num_boxes, embedding_size, padding_idx=self.num_boxes-1) # Location parameter value for the upper bound gumbel random variable (center embedding)
        self.u_context_lower_bound = nn.Embedding(self.num_boxes, embedding_size, padding_idx=self.num_boxes-1) # Location parameter value for the lower bound gumbel random variable (context embedding)
        self.u_context_upper_bound = nn.Embedding(self.num_boxes, embedding_size, padding_idx=self.num_boxes-1) # Location parameter value for the upper bound gumbel random variable (context embedding)

        # Boxes initialization
        offset = 1e-7 # offset from edges (the distance of a dimension edge from the values 0 and 1 will always be greater than the offset)
        dimension_width = 0.1
        min_value = offset
        max_value = 1 - offset - dimension_width
        self._initialize_boxes(self.num_boxes, embedding_size, min_value, max_value, dimension_width)
    
    def _gumball_soft_max(self, vector1, vector2, beta=1):
        ans = beta * torch.logaddexp(vector1/beta, vector2/beta)
        ans = torch.max(ans, torch.max(vector1,vector2)) # This line is for stability. We could get a nan from the above line when the term inside the brackets is zero.
        return ans 
    
    def _gumball_soft_min(self, vector1, vector2, beta=1):
        ans = -beta * torch.logaddexp(-vector1/beta, -vector2/beta) 
        ans = torch.min(ans, torch.min(vector1,vector2)) # This line is for stability. We could get a nan from the above line when the term inside the brackets is zero.
        return ans

    def _intersection(self, box1_lower_bound, box1_upper_bound, box2_lower_bound, box2_upper_bound, beta=1):
        # This is a point wise function, will work for one box or a batch of boxes
        new_lower_bound = self._gumball_soft_max(box1_lower_bound, box2_lower_bound, beta=beta)
        new_upper_bound = self._gumball_soft_min(box1_upper_bound, box2_upper_bound, beta=beta)
        return new_lower_bound, new_upper_bound

    def _log_volume(self, box_lower_bound, box_upper_bound, beta=1): # we get log_volume instead since as the volume is proportional to x^d where x<1, and this can can become a very small value for large number of dimensions.
        # This function is independent of the number of boxes passed. For an input with shape (batch_size, num_boxes) it will return (batch_size, 1), volume of each box.
        return torch.sum(
            torch.log(
                F.softplus(box_upper_bound - box_lower_bound - 2 * EULER_GAMMA * beta ,beta=1/beta) + SMALL_POSITIVE_FLOAT_FOR_STABILITY
                )
            , dim = -1
        )

    def forward(self, x): # x is of the form (batch_size, x) where the x values are (center_word_index, num_negative_examples, context_words) = 1 + 5 + 10 = 16

        # those two lines does not affect non-negative indices and they change the -1 to the last box index.
        x += self.num_boxes
        x %= self.num_boxes 

        center_lower_bound = self.u_center_lower_bound(x[:, :1]) # (batch_size, 1, embedding_size)
        center_upper_bound = self.u_center_upper_bound(x[:, :1]) # (batch_size, 1, embedding_size)
        negative_lower_bound = self.u_center_lower_bound(x[:, 1: 1 + self.num_negative_examples]) # (batch_size, 5, embedding_size)
        negative_upper_bound = self.u_center_upper_bound(x[:, 1: 1 + self.num_negative_examples]) # (batch_size, 5, embedding_size)
        context_lower_bound = self.u_context_lower_bound(x[:, 1 + self.num_negative_examples:])   # (batch_size, 10, embedding_size)
        context_upper_bound = self.u_context_upper_bound(x[:, 1 + self.num_negative_examples:])   # (batch_size, 10, embedding_size)

        context_intersection_lower_bound, context_intersection_upper_bound = context_lower_bound[:, :1], context_upper_bound[:, :1]
        for i in range(1,context_lower_bound.shape[1]):
            context_intersection_lower_bound, context_intersection_upper_bound = self._intersection(
                context_intersection_lower_bound, context_intersection_upper_bound, context_lower_bound[:, i:i+1], context_upper_bound[:, i:i+1]) 
       
        positive_intersection_lower_bound, positive_intersection_upper_bound = self._intersection(
            center_lower_bound, center_upper_bound, context_intersection_lower_bound,context_intersection_upper_bound)
        
        negative_intersection_lower_bound, negative_intersection_upper_bound = self._intersection(
            negative_lower_bound, negative_upper_bound, context_intersection_lower_bound,context_intersection_upper_bound)

        positive_score = self._log_volume(positive_intersection_lower_bound, positive_intersection_upper_bound)
        negative_score = self._log_volume(negative_intersection_lower_bound, negative_intersection_upper_bound)

        return positive_score, negative_score

    def get_similarity(self, word1, word2, condition_on_first_word=False):
        with torch.no_grad():
            word1, word2 = torch.tensor(word1), torch.tensor(word2)
            intersection_lower_bound, intersection_upper_bound = self._intersection(self.u_center_lower_bound(word1), self.u_center_upper_bound(word1), self.u_center_lower_bound(word2), self.u_center_upper_bound(word2))
            score = self._log_volume(intersection_lower_bound,intersection_upper_bound)
            if condition_on_first_word:
                score -= self._log_volume(self.u_center_lower_bound(word1), self.u_center_upper_bound(word1))
            return score.item()
        





class Word2EllipsoidCBOW(nn.Module):

    def __init__(self, vocab_size, embedding_size, num_negative_examples):
        super().__init__()

        self.num_regions = vocab_size + 1 # We add a unity region at the end.
        self.num_negative_examples = num_negative_examples

        self.center_mean = nn.Embedding(self.num_regions,embedding_size, padding_idx=self.num_regions-1)        # the mean parameter value for a particular gaussian distribution
        self.center_pre_variance = nn.Embedding(self.num_regions,embedding_size, padding_idx=self.num_regions-1)    # the log of the variance parameter value for a particular gaussian distribution
        self.center_constant = nn.Embedding(self.num_regions,1,padding_idx=self.num_regions-1)

        self.context_mean = nn.Embedding(self.num_regions,embedding_size, padding_idx=self.num_regions-1)       # the mean parameter value for a particular gaussian distribution
        self.context_pre_variance = nn.Embedding(self.num_regions,embedding_size, padding_idx=self.num_regions-1)   # the log of the variance parameter value for a particular gaussian distribution  
        self.context_constant = nn.Embedding(self.num_regions,1,padding_idx=self.num_regions-1)

        # Custom Initialization 
        self.center_mean.weight.data.uniform_(-1,1)
        self.context_mean.weight.data.uniform_(-1,1)

        self.center_pre_variance.weight.data.uniform_(-1, 0) # so that the variance value is intalized between exp(-1) and 1
        self.context_pre_variance.weight.data.uniform_(-1,0)

        # Set constant value to zero and make non-learnable
        self.center_constant.weight.data.zero_()
        self.center_constant.weight.requires_grad = False

        self.context_constant.weight.data.zero_()
        self.context_constant.weight.requires_grad = False
        

        # Initialize the unity region
        # The following intializition is working for ADAMW because it adds the weight decay term directly to the weights using the equation w(t) = w(t-1) - lambda*w(t-1). This will not affect a weight with either zero or inf values. This is different in Adam or other optimizers.
        self.center_mean.weight[self.num_regions - 1].data.zero_()
        self.center_pre_variance.weight[self.num_regions - 1].data.fill_(torch.inf)

        self.context_mean.weight[self.num_regions - 1].data.zero_()
        self.context_pre_variance.weight[self.num_regions - 1].data.fill_(torch.inf)

    def _intersection(self, region1_mean, region1_variance, region1_constant, region2_mean, region2_variance, region2_constant):
        new_variance = 1/ (1/region1_variance + 1/region2_variance)
        new_mean = new_variance * (region1_mean/region1_variance + region2_mean/region2_variance)
        new_constant = region1_constant + region2_constant - 0.5 * torch.sum( (region1_mean-region2_mean)**2 / (region1_variance + region2_variance), dim=-1, keepdim=True)
        return new_mean, new_variance, new_constant

    def _log_volume(self, region_variance, region_constant): # we get log_volume instead since as the volume is proportional to x^d where x<1, and this can can become a very small value for large number of dimensions.
        # This function is independent of the number of regions passed. For an input with shape (batch_size, num_regions) it will return (batch_size, 1), volume of each region.  
        return torch.sum(
            0.5 * torch.log( 2 * torch.pi * region_variance + SMALL_POSITIVE_FLOAT_FOR_STABILITY), dim=-1, keepdim=True
            ) + region_constant

    def forward(self, x): # x is of the form (batch_size, x) where the x values are (center_word_index, num_negative_examples, context_words) = 1 + 5 + 10 = 16

        # those two lines does not affect non-negative indices and they change the -1 to the last region index.
        x += self.num_regions
        x %= self.num_regions 

        center_mean = self.center_mean(x[:, :1]) # (batch_size, 1, embedding_size)
        center_pre_variance = self.center_pre_variance(x[:, :1]) # (batch_size, 1, embedding_size)
        center_constant = self.center_constant(x[:, :1])

        negative_mean = self.center_mean(x[:, 1: 1 + self.num_negative_examples]) # (batch_size, 5, embedding_size)
        negative_pre_variance = self.center_pre_variance(x[:, 1: 1 + self.num_negative_examples]) # (batch_size, 5, embedding_size)
        negative_constant = self.center_constant(x[:, 1: 1 + self.num_negative_examples])

        context_mean = self.context_mean(x[:, 1 + self.num_negative_examples:])   # (batch_size, 10, embedding_size)
        context_pre_variance = self.context_pre_variance(x[:, 1 + self.num_negative_examples:])   # (batch_size, 10, embedding_size)
        context_constant = self.context_constant(x[:, 1 + self.num_negative_examples:]) 

        center_variance = F.softplus(center_pre_variance)
        negative_variance = F.softplus(negative_pre_variance)
        context_variance = F.softplus(context_pre_variance)

        context_intersection_mean, context_intersection_variance, context_intersection_constant = context_mean[:, :1], context_variance[:, :1], context_constant[:, :1]
        for i in range(1,context_mean.shape[1]):

            context_intersection_mean, context_intersection_variance, context_intersection_constant = self._intersection(
                context_intersection_mean, context_intersection_variance, context_intersection_constant, context_mean[:, i:i+1], context_variance[:, i:i+1], context_constant[:, i:i+1]
                )
      
        positive_intersection_mean, positive_intersection_variance, positive_intersection_constant = self._intersection(
            center_mean, center_variance, center_constant, context_intersection_mean, context_intersection_variance, context_intersection_constant)

        negative_intersection_mean, negative_intersection_variance, negative_intersection_constant = self._intersection(
            negative_mean, negative_variance, negative_constant, context_intersection_mean, context_intersection_variance, context_intersection_constant)
        
        positive_score = self._log_volume(positive_intersection_variance, positive_intersection_constant)
        negative_score = self._log_volume(negative_intersection_variance, negative_intersection_constant)

        return positive_score, negative_score

    def get_similarity(self, word1, word2, condition_on_first_word=False):
        with torch.no_grad():
            word1, word2 = torch.tensor(word1), torch.tensor(word2)

            word1_mean = self.center_mean(word1)
            word1_pre_variance = self.center_pre_variance(word1)
            word1_constant = self.center_constant(word1)

            word2_mean = self.center_mean(word2)
            word2_pre_variance = self.center_pre_variance(word2)
            word2_constant = self.center_constant(word2)

            word1_variance = F.softplus(word1_pre_variance)
            word2_variance = F.softplus(word2_pre_variance)

            intersection_mean, intersection_variance, intersection_constant = self._intersection(word1_mean, word1_variance, word1_constant, word2_mean, word2_variance, word2_constant)

            score = self._log_volume(intersection_variance, intersection_constant)

            if condition_on_first_word:
                score -= self._log_volume(word1_variance, word1_constant)

            return score.item()


###################################### SKIPGRAM STYLE MODELS



class Word2VecSkipGram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Word2BoxSkipGram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Word2EllipsoidSkipGram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)