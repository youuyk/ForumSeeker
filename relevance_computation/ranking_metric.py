from collections import OrderedDict
from itertools import repeat 

def mrr(rankings):
    
    reverse_rankings = []
    for r in rankings:
        
        reverse_rankings.append(1/r)
    reverse_rankings = sum(reverse_rankings)
    return reverse_rankings/len(rankings)

def make_ranking_result_dict(resultDict):
    
    ranking_dict= {}
    ranking = 0 
    for url, value in resultDict.items():
        if ranking == 0:
            last_value = value 
            ranking = ranking + 1 
            n = 1 
        else:
            if last_value != value:
                ranking = n + 1 
                n = ranking 
                last_value = value 
            else:
                n = n + 1 
        ranking_dict[url] = ranking 
        
    return ranking_dict

def get_diff(max_value, value):
    return max_value - value

def get_diff_normalize(min_value, max_value, value):
    
    norm_diff = (value - min_value) / (max_value - min_value)
    return norm_diff

class make_ranking_dict:
    
    def make_ranking_dict_type(resultDict):
        
        modelIndex = resultDict['modelIndex']
        result = resultDict['resultDict']
        if result == None:
            return {'modelIndex': modelIndex, 'rankingDict': None}
        
        #print("4. Making rankingDict (type 0)")
        ranking_dict = make_ranking_result_dict(result)
        return {'modelIndex': modelIndex, 'rankingDict': ranking_dict}
        
