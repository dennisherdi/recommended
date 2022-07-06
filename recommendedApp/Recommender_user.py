from sqlalchemy.sql.expression import update
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String , update
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import sklearn
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def recomended_user(request):
    engine = create_engine(
        "mysql://admin:db_dev_Interconnect_Data_Project@interconnect-data-project-dev.cqorlseuayk8.ap-southeast-1.rds.amazonaws.com/interconnect_data",
        connect_args={'connect_timeout': 30}, echo=False
    )
    conn = engine.connect()

    df = pd.read_sql("select id_user, id_org, sum(Detail+Connects+Track+CreateNote+SeeMore+ReplyNote) - sum(Disconnects+Untrack+SeeLess+DeleteNote)  sum_p from (select id_user,id_org, sum(name_action = 'Detail') as Detail, sum(name_action = 'Connect') as Connects, sum(name_action = 'Track') as Track, sum(name_action = 'Disconnect') as Disconnects, sum(name_action = 'Untrack') as Untrack, sum(name_action = 'Create Note') as CreateNote, sum(name_action = 'Delete Note') as DeleteNote, sum(name_action = 'See Less') as SeeLess, sum(name_action = 'See More') as SeeMore, sum(name_action = 'Reply Note') as ReplyNote from ic_log_organizations group by id_user, id_org) a group by id_user,id_org", conn)

    df.dropna(inplace = True) 

    #Creating a sparse pivot table with users in rows and items in columns
    users_items_pivot_matrix_df = df.pivot(index='id_user', 
                                        columns='id_org',
                                        values='sum_p').fillna(0)

    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    users_ids = list(users_items_pivot_matrix_df.index)
    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

    #The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 15
    #Performs matrix factorization of the original user item matrix
    #U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()

    class CFRecommender:
    
        MODEL_NAME = 'Collaborative Filtering'
    
        def __init__(self, cf_predictions_df, items_df=None):
            self.cf_predictions_df = cf_predictions_df
            self.items_df = items_df
        
        def get_model_name(self):
            return self.MODEL_NAME
        
        def recommend_items(self, id_user, items_to_ignore=[], topn=10, verbose=False):
            # Get and sort the user's predictions
            sorted_user_predictions = self.cf_predictions_df[id_user].sort_values(ascending=False) \
                                    .reset_index().rename(columns={id_user: 'sum_p'})

            # Recommend the highest predicted rating movies that the user hasn't seen yet.
            recommendations_df = sorted_user_predictions[~sorted_user_predictions['id_org'].isin(items_to_ignore)] \
                               .sort_values('sum_p', ascending = False) \
                               .head(topn)

            return recommendations_df
    
    cf_recommender_model = CFRecommender(cf_preds_df, df)

    id_user = str(request.POST["id"])

    Recommend = cf_recommender_model.recommend_items(int(id_user), topn=10)
    Recommend = Recommend.to_dict('r')

    respond = {
        "status" : True,
        "message": "succes",
        'content': Recommend
    }
    return JsonResponse(respond, safe=False)    