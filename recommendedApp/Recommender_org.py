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
def recomended_org(request):
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

    X = users_items_pivot_matrix_df.T
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    id_org = str(request.POST["id"])
    company_names = list(X.index)
    company_ID = company_names.index(int(id_org))
    correlation_company_ID = correlation_matrix[company_ID]
    Recommend = list(X.index[correlation_company_ID > 0.90])
    Recommend.remove(int(id_org)) 
    Recommend[0:9]

    respond = {
        "status" : True,
        "message": "succes",
        'content': Recommend
    }
    return JsonResponse(respond, safe=False)  