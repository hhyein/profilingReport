from flask import Flask, render_template, request, redirect, flash
from nl4dv import NL4DV

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
import numpy as np
import joblib
import json
import os

import imp_list.imputation as imputation
import imp_list.statistics as statistics
import imp_list.tree as tree

import temp_list.div as div
import temp_list.svg as svg

recommend_method = ''
recommend_type = ''

current_file = ''
cnt = 2

cnt_create = 0
create_div = ''
create_vlSpec = ''
create_bar = ''
create_heatmap = ''
create_histogram = ''
create_density = ''
create_ecdf = ''

app = Flask(__name__)
app.secret_key = "vislab"
application = Flask(import_name = __name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/profiling_vis')
    
@app.route('/profiling_vis', methods=['GET', 'POST'])
def profiling_vis():
    with open('static/data/tree_data.json') as json_file:
        tree_data = json.load(json_file)

    root = tree.TreeNode(tree_data['file'], name = tree_data['name'], state = tree_data['state'], action = tree_data['action'])
    root = root.dict_to_tree(tree_data['children'])

    global current_file
    current_node = root.find_state('current')
    current_file = 'static/data/' + str(current_node.file) + '.csv'
    
    return render_template("index.html", tree_data = tree_data, create_div = create_div, create_vlSpec = create_vlSpec, create_bar = create_bar, create_heatmap = create_heatmap, create_histogram = create_histogram, create_density = create_density, create_ecdf = create_ecdf)

@app.route('/setting', methods=['GET', 'POST'])
def setting():
    global recommend_method, recommend_type
    try:
        recommend_method = request.form["first_radio_name"]
        recommend_type = request.form["second_radio_name"]
    except:
        recommend_type = ''

    return redirect('/profiling_vis')

@app.route('/input_query', methods=['GET', 'POST'])
def input_query():
    global current_file, cnt_create, create_div, create_vlSpec, create_bar
    file_name = current_file
    df = pd.read_csv(file_name)

    nl4dv_df = df.dropna()
    nl4dv_df = nl4dv_df.to_dict('records')
    nl4dv_instance = NL4DV(data_url = os.path.join(file_name))
    dependency_parser_config = {"name": "spacy", "model": "en_core_web_sm", "parser": None}
    nl4dv_instance.set_dependency_parser(config = dependency_parser_config)

    result = request.form
    query = result['input_query']
    nl4dv_output = nl4dv_instance.analyze_query(query)

    # extraction attribute, task, vistype
    try:
        attributes = nl4dv_output['visList'][0]['attributes']
        tasks = nl4dv_output['visList'][0]['tasks']
        visType = nl4dv_output['visList'][0]['visType']
    except:   
        flash("please writing valid query")
        return redirect('/profiling_vis')

    if type(attributes) == list:
        attributes = ",".join(attributes)
    if type(tasks) == list:
        tasks = ",".join(tasks)
    if type(visType) == list:
        visType = ",".join(visType)

    # extraction vlspec
    vlSpec = nl4dv_output['visList'][0]['vlSpec']
    vlSpec['data']['values'] = nl4dv_df

    vlSpec['width'] = "container"
    vlSpec['height'] = "container"

    # preprocessing vlspec
    if 'encoding' in vlSpec:
        if 'x' in vlSpec['encoding']:
            if 'aggregate' in vlSpec['encoding']['x']:
                del vlSpec['encoding']['x']['aggregate']
    if 'encoding' in vlSpec:
        if 'y' in vlSpec['encoding']:
            if 'aggregate' in vlSpec['encoding']['y']:
                del vlSpec['encoding']['y']['aggregate']
    if 'encoding' in vlSpec:
        if 'x' in vlSpec['encoding']:
            if 'bin' in vlSpec['encoding']['x']:
                del vlSpec['encoding']['x']['bin']
    if 'encoding' in vlSpec:
        if 'color' in vlSpec['encoding']:
            if 'aggregate' in vlSpec['encoding']['color']:
                del vlSpec['encoding']['color']['aggregate']

    del vlSpec['mark']['tooltip']
    del vlSpec['data']['format']
    del vlSpec['data']['url']

    # generation div
    create_div = create_div + div.input_query_div(cnt_create, attributes, tasks, visType)

    # vlSpec
    create_vlSpec = create_vlSpec + svg.svg_vlSpec(cnt_create, vlSpec)

    cnt_create = cnt_create + 1

    return redirect('/profiling_vis')

@app.route('/current', methods=['GET', 'POST'])
def current():
    data = request.get_data().decode('utf-8').split('&')
    file_name = 'static/data/' + data[0][10:] + '.csv'
    name = data[1][5:]

    df = pd.read_csv(file_name)
    df = df[sorted(df.columns)]
    column_list = list(df)

    # tree
    with open('static/data/tree_data.json') as json_file:
        tree_data = json.load(json_file)
    root = tree.TreeNode(file = tree_data['file'], name = tree_data['name'], state = tree_data['state'], action = tree_data['action'])
    root = root.dict_to_tree(tree_data['children'])

    # find children node
    my_node = root.find_name(name)
    children_node = my_node.children

    global recommend_method, cnt_create, create_div, create_bar
    if recommend_method == 'A':
        # current df quality issue
        missing, extreme, total = statistics.quality_issue(df)
        current_df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        current_df_problem = current_df_problem.sort_values(by = 'index').reset_index(drop = True)
        
        current_df_problem = current_df_problem.drop(['index'], axis = 1)
        current_df_problem = current_df_problem.sum().tolist()

        # compare current - children quality issue to generation div
        action = []
        output_value = [[], [], []]
        output_percent = [[], [], []]
        output_sign = [[], [], []]

        for i in range(0, 3):
            action.append(children_node[i].action)
            
            child_df = pd.read_csv('static/data/' + children_node[i].file + '.csv')
            child_df = child_df[sorted(child_df.columns)]
            
            missing, extreme, total = statistics.quality_issue(child_df)
            child_df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
            child_df_problem = child_df_problem.sort_values(by = 'index').reset_index(drop = True)
            child_df_problem = child_df_problem.drop(['index'], axis = 1)
            
            child_df_problem = child_df_problem.sum().tolist()

            for j in range(len(current_df_problem)):
                diff = current_df_problem[j] - child_df_problem[j]

                if diff > 0:
                    diff_percent = (100 * diff)/current_df_problem[j]
                    output_sign[i].append('decrease')
                elif diff == 0:
                    diff_percent = 0
                    output_sign[i].append('decrease')
                else:
                    diff = -diff
                    diff_percent = 0
                    output_sign[i].append('increase')

                output_value[i].append(diff)
                output_percent[i].append(diff_percent)

        # generation div
        create_div = create_div + div.current_div_A(cnt_create, name, children_node, action, output_value, output_percent, output_sign)

        # barchart - each
        missing, extreme, total = statistics.quality_issue(df)
        max_value = max(total)

        bar_output_df = pd.DataFrame({x for x in zip(column_list, missing, extreme)}, columns = ['index', 'missing', 'extreme'])
        bar_output_df = bar_output_df.sort_values(by = 'index').reset_index(drop = True)

        bar_output_df.to_csv("static/data/bar_" + str(cnt_create) + ".csv", index = False)
        create_bar = create_bar + svg.svg_bar1(cnt_create, max_value)

        # barchart - total
        create_bar = create_bar + svg.svg_bar2(cnt_create, max_value)

    if recommend_method == 'B':
        # current df quality metric
        output = statistics.quality_metric_total(df)
        current_df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        current_df_problem = current_df_problem.sort_values(by = 'index').reset_index(drop = True)

        current_df_problem['skewness'] = current_df_problem['skewness'].abs()
        current_df_problem['kurtosis'] = current_df_problem['kurtosis'].abs()

        # select the column with the lowest quality
        target_column = current_df_problem.iloc[current_df_problem[recommend_type].idxmax()]['index']
        target_idx = df.columns.get_loc(target_column)

        current_df_problem = current_df_problem.iloc[target_idx, :]
        current_df_problem = current_df_problem.tolist()
        del current_df_problem[0]

        # compare current - children problem to generation div
        action = []
        output_value = [[], [], []]
        output_percent = [[], [], []]
        output_sign = [[], [], []]

        for i in range(0, 3):
            action.append(children_node[i].action)
            
            child_df = pd.read_csv('static/data/' + children_node[i].file + '.csv')
            child_df = child_df[sorted(child_df.columns)]
            
            output = statistics.quality_metric_total(child_df)
            child_df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
            child_df_problem = child_df_problem.sort_values(by = 'index').reset_index(drop = True)

            child_df_problem['skewness'] = child_df_problem['skewness'].abs()
            child_df_problem['kurtosis'] = child_df_problem['kurtosis'].abs()

            child_df_problem = child_df_problem.iloc[target_idx, :]
            child_df_problem = child_df_problem.tolist()
            del child_df_problem[0]

            for j in range(len(current_df_problem)):
                diff = current_df_problem[j] - child_df_problem[j]
                diff = round(diff, 5)

                if diff > 0:
                    diff_percent = (100 * diff)/current_df_problem[j]
                    output_sign[i].append('decrease')
                elif diff == 0:
                    diff_percent = 0
                    output_sign[i].append('decrease')
                else:
                    diff = -diff
                    diff_percent = 0
                    output_sign[i].append('increase')

                output_value[i].append(diff)
                output_percent[i].append(diff_percent)

        # generation div
        create_div = create_div + div.current_div_B(cnt_create, name, children_node, action, output_value, output_percent, output_sign)

        # barchart - each
        output = statistics.quality_metric_total(df)
        bar_output_df = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        bar_output_df = bar_output_df.sort_values(by = 'index').reset_index(drop = True)

        bar_output_df['skewness'] = bar_output_df['skewness'].abs()
        bar_output_df['kurtosis'] = bar_output_df['kurtosis'].abs()

        # normalization
        norm_df = bar_output_df.drop(['index'], axis = 1)
        column_df = bar_output_df['index']

        scaler = MinMaxScaler()
        norm_df = scaler.fit_transform(norm_df)
        norm_df = pd.DataFrame(norm_df)

        norm_df = norm_df.rename(columns = {0: 'kstest', 1: 'skewness', 2: 'kurtosis'})
        norm_df = pd.concat([column_df, norm_df], axis = 1)
        norm_df.to_csv("static/data/bar1_" + str(cnt_create) + ".csv", index = False)

        norm_df = norm_df.drop(['index'], axis = 1)
        max_value = norm_df.melt().value.max()
        create_bar = create_bar + svg.svg_bar3(cnt_create, max_value, 'bar1')

        # barchart - total
        bar_output_df.to_csv("static/data/bar2_" + str(cnt_create) + ".csv", index = False)
        max_value = bar_output_df[recommend_type].max()
        create_bar = create_bar + svg.svg_bar4(cnt_create, max_value, recommend_type)

    if recommend_method == 'C':
        # compare current - children quality issue to generation div
        action = []
        output_value = [[], [], [], [], [], []]
        output_percent = [[], [], [], [], [], []]
        output_sign = [[], [], [], [], [], []]

        # current df quality issue
        missing, extreme, total = statistics.quality_issue(df)
        current_df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        current_df_problem = current_df_problem.sort_values(by = 'index').reset_index(drop = True)
        
        current_df_problem = current_df_problem.drop(['index'], axis = 1)
        current_df_problem = current_df_problem.sum().tolist()

        for i in range(0, 3):
            action.append(children_node[i].action)
            
            child_df = pd.read_csv('static/data/' + children_node[i].file + '.csv')
            child_df = child_df[sorted(child_df.columns)]
            
            missing, extreme, total = statistics.quality_issue(child_df)
            child_df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
            child_df_problem = child_df_problem.sort_values(by = 'index').reset_index(drop = True)
            child_df_problem = child_df_problem.drop(['index'], axis = 1)
            
            child_df_problem = child_df_problem.sum().tolist()

            for j in range(len(current_df_problem)):
                diff = current_df_problem[j] - child_df_problem[j]

                if diff > 0:
                    diff_percent = (100 * diff)/current_df_problem[j]
                    output_sign[i].append('decrease')
                elif diff == 0:
                    diff_percent = 0
                    output_sign[i].append('decrease')
                else:
                    diff = -diff
                    diff_percent = 0
                    output_sign[i].append('increase')

                output_value[i].append(diff)
                output_percent[i].append(diff_percent)

        # current df quality metric
        output = statistics.quality_metric_total(df)
        current_df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        current_df_problem = current_df_problem.sort_values(by = 'index').reset_index(drop = True)

        current_df_problem['skewness'] = current_df_problem['skewness'].abs()
        current_df_problem['kurtosis'] = current_df_problem['kurtosis'].abs()

        # select the column with the lowest quality
        ##### example - skewness
        target_column = current_df_problem.iloc[current_df_problem['skewness'].idxmax()]['index']
        target_idx = df.columns.get_loc(target_column)

        current_df_problem = current_df_problem.iloc[target_idx, :]
        current_df_problem = current_df_problem.tolist()
        del current_df_problem[0]

        for i in range(3, 6):
            # quality metric
            output = statistics.quality_metric_total(child_df)
            child_df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
            child_df_problem = child_df_problem.sort_values(by = 'index').reset_index(drop = True)

            child_df_problem['skewness'] = child_df_problem['skewness'].abs()
            child_df_problem['kurtosis'] = child_df_problem['kurtosis'].abs()

            child_df_problem = child_df_problem.iloc[target_idx, :]
            child_df_problem = child_df_problem.tolist()
            del child_df_problem[0]

            for j in range(len(current_df_problem)):
                diff = current_df_problem[j] - child_df_problem[j]
                diff = round(diff, 5)

                if diff > 0:
                    diff_percent = (100 * diff)/current_df_problem[j]
                    output_sign[i].append('decrease')
                elif diff == 0:
                    diff_percent = 0
                    output_sign[i].append('decrease')
                else:
                    diff = -diff
                    diff_percent = 0
                    output_sign[i].append('increase')

                output_value[i].append(diff)
                output_percent[i].append(diff_percent)

        # generation div
        create_div = create_div + div.current_div_C(cnt_create, name, children_node, action, output_value, output_percent, output_sign)

        # barchart - quality issue
        missing, extreme, total = statistics.quality_issue(df)
        max_value = max(total)

        bar_output_df = pd.DataFrame({x for x in zip(column_list, missing, extreme)}, columns = ['index', 'missing', 'extreme'])
        bar_output_df = bar_output_df.sort_values(by = 'index').reset_index(drop = True)

        bar_output_df.to_csv("static/data/bar_" + str(cnt_create) + ".csv", index = False)
        create_bar = create_bar + svg.svg_bar1(cnt_create, max_value)

        # barchart - quality metric
        output = statistics.quality_metric_total(df)
        bar_output_df = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        bar_output_df = bar_output_df.sort_values(by = 'index').reset_index(drop = True)

        bar_output_df['skewness'] = bar_output_df['skewness'].abs()
        bar_output_df['kurtosis'] = bar_output_df['kurtosis'].abs()

        # normalization
        norm_df = bar_output_df.drop(['index'], axis = 1)
        column_df = bar_output_df['index']

        scaler = MinMaxScaler()
        norm_df = scaler.fit_transform(norm_df)
        norm_df = pd.DataFrame(norm_df)

        norm_df = norm_df.rename(columns = {0: 'kstest', 1: 'skewness', 2: 'kurtosis'})
        norm_df = pd.concat([column_df, norm_df], axis = 1)
        norm_df.to_csv("static/data/bar1_" + str(cnt_create) + ".csv", index = False)

        norm_df = norm_df.drop(['index'], axis = 1)
        max_value = norm_df.melt().value.max()
        create_bar = create_bar + svg.svg_bar3(cnt_create, max_value, 'bar2')

    cnt_create = cnt_create + 1

    return redirect('/profiling_vis')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = request.get_data().decode('utf-8').split('&')
    file_name = 'static/data/' + data[0][10:] + '.csv'
    name = data[1][5:]

    df = pd.read_csv(file_name)
    df = df[sorted(df.columns)]
    column_list = list(df)

    global recommend_method, current_file, cnt_create, create_div, create_heatmap, create_histogram, create_density, create_ecdf
    current_df = pd.read_csv(current_file)
    current_df = current_df[sorted(df.columns)]

    if recommend_method == 'A':
        # generation div
        create_div = create_div + div.recommend_div_A(cnt_create, name)

        missing, extreme, total = statistics.quality_issue(current_df)
        df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)
        target_column = df_problem.iloc[df_problem['total'].idxmax()]['index']

        # heatmap - current
        heatmap_output_df, y_list = svg.make_heatmap_df(column_list, current_df, current_df)

        heatmap_output_df.to_csv("static/data/heatmap1_" + str(cnt_create) + ".csv", index = False)
        create_heatmap = create_heatmap + svg.svg_heatmap(cnt_create, column_list, y_list, target_column, 1, 'heatmap_before')

        # heatmap - after
        heatmap_output_df, temp = svg.make_heatmap_df(column_list, current_df, df)

        heatmap_output_df.to_csv("static/data/heatmap2_" + str(cnt_create) + ".csv", index = False)
        create_heatmap = create_heatmap + svg.svg_heatmap(cnt_create, column_list, y_list, target_column, 2, 'heatmap_after')

        # histogram
        current_output_df = current_df.dropna()
        min_value = current_output_df[target_column].min()
        max_value = current_output_df[target_column].max()
        lower, upper = statistics.lower_upper(current_output_df, target_column)

        current_output_df.to_csv("static/data/histogram1_" + str(cnt_create) + ".csv", index = False)
        create_histogram = create_histogram + svg.svg_histogram1(cnt_create, target_column, min_value, max_value, lower, upper, 1, 'histogram_before')

        df.to_csv("static/data/histogram2_" + str(cnt_create) + ".csv", index = False)
        create_histogram = create_histogram + svg.svg_histogram1(cnt_create, target_column, min_value, max_value, lower, upper, 2, 'histogram_after')

    if recommend_method == 'B':
        # generation div
        create_div = create_div + div.recommend_div_B(cnt_create, name)

        output = statistics.quality_metric_total(current_df)
        df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)

        df_problem['skewness'] = df_problem['skewness'].abs()
        df_problem['kurtosis'] = df_problem['kurtosis'].abs()

        target_column = df_problem.iloc[df_problem['skewness'].idxmax()]['index']

        # density - current
        density_df = current_df.dropna()
        density_df = density_df[[target_column]]

        # density - after
        density_after_df = df
        density_after_df = density_after_df[[target_column]]

        # density - normal
        density_origin_df = pd.read_csv('static/data/missing.csv').dropna()
        density_origin_df = density_origin_df[[target_column]]

        mu = density_origin_df.mean()
        std = density_origin_df.std()

        rv = stats.norm(loc = mu, scale = std)
        x = pd.DataFrame(rv.rvs(size = 5000, random_state = 0))

        min_value = min([density_origin_df.values.min(), density_df.values.min(), density_after_df.values.min(), x.values.min()])
        max_value = max([density_origin_df.values.max(), density_df.values.max(), density_after_df.values.max(), x.values.max()])

        # normal - current
        density_output_df = svg.make_density_df(x, density_df, 'normal', 'current')
        density_output_df.to_csv("static/data/density2_" + str(cnt_create) + ".csv", index = False)
        create_density = create_density + svg.svg_density(cnt_create, 'skewness', min_value, max_value, 2, 'current', 'density_before')

        # normal - after
        density_output_df = svg.make_density_df(x, density_after_df, 'normal', 'after')
        density_output_df.to_csv("static/data/density3_" + str(cnt_create) + ".csv", index = False)
        create_density = create_density + svg.svg_density(cnt_create, 'skewness', min_value, max_value, 3, 'after', 'density_after')

        # ecdf - current
        ecdf_df = current_df[target_column].dropna()
        current_output_df = svg.make_ecdf_df(ecdf_df, 'kstest2')

        # ecdf - after
        ecdf_after_df = df[target_column].dropna()
        after_output_df = svg.make_ecdf_df(ecdf_after_df, 'kstest2')

        # ecdf - normal
        ecdf_origin_df = pd.read_csv('static/data/missing.csv')
        ecdf_origin_df = ecdf_origin_df[target_column].dropna()

        mu = ecdf_origin_df.mean()
        std = ecdf_origin_df.std()

        rv = stats.norm(loc = mu, scale = std)
        x = rv.rvs(size = 5000, random_state = 0)
        x = svg.make_ecdf_df(x, 'kstest1')

        # normal - current
        ecdf_output_df = pd.concat([x, current_output_df])
        ecdf_output_df.to_csv("static/data/ecdf2_" + str(cnt_create) + ".csv", index = False)
        create_ecdf = create_ecdf + svg.svg_ecdf(cnt_create, 2, 'current', 'ecdf_before')

        # normal - after
        ecdf_output_df = pd.concat([x, after_output_df])
        ecdf_output_df.to_csv("static/data/ecdf3_" + str(cnt_create) + ".csv", index = False)
        create_ecdf = create_ecdf + svg.svg_ecdf(cnt_create, 3, 'after', 'ecdf_after')

    if recommend_method == 'C':
        # generation div
        create_div = create_div + div.recommend_div_A(cnt_create, name)

        missing, extreme, total = statistics.quality_issue(current_df)
        df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)
        target_column = df_problem.iloc[df_problem['total'].idxmax()]['index']

        # heatmap - current
        heatmap_output_df, y_list = svg.make_heatmap_df(column_list, current_df, current_df)

        heatmap_output_df.to_csv("static/data/heatmap1_" + str(cnt_create) + ".csv", index = False)
        create_heatmap = create_heatmap + svg.svg_heatmap(cnt_create, column_list, y_list, target_column, 1, 'heatmap_before')

        # heatmap - after
        heatmap_output_df, temp = svg.make_heatmap_df(column_list, current_df, df)

        heatmap_output_df.to_csv("static/data/heatmap2_" + str(cnt_create) + ".csv", index = False)
        create_heatmap = create_heatmap + svg.svg_heatmap(cnt_create, column_list, y_list, target_column, 2, 'heatmap_after')

        # histogram
        current_output_df = current_df.dropna()
        min_value = current_output_df[target_column].min()
        max_value = current_output_df[target_column].max()
        lower, upper = statistics.lower_upper(current_output_df, target_column)

        current_output_df.to_csv("static/data/histogram1_" + str(cnt_create) + ".csv", index = False)
        create_histogram = create_histogram + svg.svg_histogram1(cnt_create, target_column, min_value, max_value, lower, upper, 1, 'histogram_before')

        df.to_csv("static/data/histogram2_" + str(cnt_create) + ".csv", index = False)
        create_histogram = create_histogram + svg.svg_histogram1(cnt_create, target_column, min_value, max_value, lower, upper, 2, 'histogram_after')

        cnt_create = cnt_create + 1

        # generation div
        create_div = create_div + div.recommend_div_B(cnt_create, name)

        output = statistics.quality_metric_total(current_df)
        df_problem = pd.DataFrame({x for x in zip(column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)

        df_problem['skewness'] = df_problem['skewness'].abs()
        df_problem['kurtosis'] = df_problem['kurtosis'].abs()

        target_column = df_problem.iloc[df_problem['skewness'].idxmax()]['index']

        # density - current
        density_df = current_df.dropna()
        density_df = density_df[[target_column]]

        # density - after
        density_after_df = df
        density_after_df = density_after_df[[target_column]]

        # density - normal
        density_origin_df = pd.read_csv('static/data/missing.csv').dropna()
        density_origin_df = density_origin_df[[target_column]]

        mu = density_origin_df.mean()
        std = density_origin_df.std()

        rv = stats.norm(loc = mu, scale = std)
        x = pd.DataFrame(rv.rvs(size = 5000, random_state = 0))

        min_value = min([density_origin_df.values.min(), density_df.values.min(), density_after_df.values.min(), x.values.min()])
        max_value = max([density_origin_df.values.max(), density_df.values.max(), density_after_df.values.max(), x.values.max()])

        # normal - current
        density_output_df = svg.make_density_df(x, density_df, 'normal', 'current')
        density_output_df.to_csv("static/data/density2_" + str(cnt_create) + ".csv", index = False)
        create_density = create_density + svg.svg_density(cnt_create, 'skewness', min_value, max_value, 2, 'current', 'density_before')

        # normal - after
        density_output_df = svg.make_density_df(x, density_after_df, 'normal', 'after')
        density_output_df.to_csv("static/data/density3_" + str(cnt_create) + ".csv", index = False)
        create_density = create_density + svg.svg_density(cnt_create, 'skewness', min_value, max_value, 3, 'after', 'density_after')

        # ecdf - current
        ecdf_df = current_df[target_column].dropna()
        current_output_df = svg.make_ecdf_df(ecdf_df, 'kstest2')

        # ecdf - after
        ecdf_after_df = df[target_column].dropna()
        after_output_df = svg.make_ecdf_df(ecdf_after_df, 'kstest2')

        # ecdf - normal
        ecdf_origin_df = pd.read_csv('static/data/missing.csv')
        ecdf_origin_df = ecdf_origin_df[target_column].dropna()

        mu = ecdf_origin_df.mean()
        std = ecdf_origin_df.std()

        rv = stats.norm(loc = mu, scale = std)
        x = rv.rvs(size = 5000, random_state = 0)
        x = svg.make_ecdf_df(x, 'kstest1')

        # normal - current
        ecdf_output_df = pd.concat([x, current_output_df])
        ecdf_output_df.to_csv("static/data/ecdf2_" + str(cnt_create) + ".csv", index = False)
        create_ecdf = create_ecdf + svg.svg_ecdf(cnt_create, 2, 'current', 'ecdf_before')

        # normal - after
        ecdf_output_df = pd.concat([x, after_output_df])
        ecdf_output_df.to_csv("static/data/ecdf3_" + str(cnt_create) + ".csv", index = False)
        create_ecdf = create_ecdf + svg.svg_ecdf(cnt_create, 3, 'after', 'ecdf_after')

    cnt_create = cnt_create + 1

    return redirect('/profiling_vis')

@app.route('/update_tree', methods=['GET', 'POST'])
def update_tree():
    data = request.get_data().decode('utf-8').split('&')
    file_name = 'static/data/' + data[0][10:] + '.csv'
    name = data[1][5:]

    df = pd.read_csv(file_name)
    df = df[sorted(df.columns)]
    column_list = list(df)

    with open('static/data/tree_data.json') as json_file:
        tree_data = json.load(json_file)

    root = tree.TreeNode(tree_data['file'], name = tree_data['name'], state = tree_data['state'], action = tree_data['action'])
    root = root.dict_to_tree(tree_data['children'])

    global recommend_method, recommend_type
    if recommend_method == 'A':
        # quality issue per column in current df
        missing, extreme, total = statistics.quality_issue(df)
        df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)

        # select the column with the most quality issue
        target_column = df_problem.iloc[df_problem['total'].idxmax()]['index']
        target_df = df.loc[:, [target_column]]
        remain_df = df.drop([target_column], axis = 1)

        try:
            # action
            # if target_column have missing value
            action = ['missing mean', 'missing median', 'missing em', 'missing locf', 'outlier mean', 'outlier median', 'outlier em', 'outlier locf', 'missing remove', 'outlier remove']
            action_df = [imputation.custom_imp_mean(target_column, target_df), imputation.custom_imp_median(target_column, target_df),
                        imputation.custom_imp_em(target_column, target_df), imputation.custom_imp_locf(target_column, target_df)]

            lower, upper = statistics.lower_upper(target_df, target_column)
            outlier_df = target_df[(target_df[target_column] < lower) | (target_df[target_column] > upper)]
            outlier_index = list(outlier_df.index)

            for i in range(0, 4):
                for j in outlier_index:
                    target_df.loc[j, target_column] = action_df[i].iloc[j][target_column]
                action_df.append(target_df)

            action_df.append('missing remove')
            action_df.append('outlier remove')

            # current df quality issue
            missing, extreme, total = statistics.quality_issue(df)
            current_df_problem = pd.DataFrame({x for x in zip(column_list, total)}, columns = ['index', 'total'])
            current_df_problem = current_df_problem.sort_values(by = 'index').reset_index(drop = True)
            current_df_problem = current_df_problem.drop(['index'], axis = 1)
            
            current_df_problem = current_df_problem.sum().tolist()

            # compare current - action
            output = []

            for i in range(0, 10):
                if action[i] == 'missing remove':
                    tmp_df = df.dropna(subset = [target_column], how = 'all')
                elif action[i] == 'outlier remove':
                    tmp_df = df.drop(outlier_df.index, axis = 0).reset_index(drop = True)
                else:
                    tmp_df = action_df[i]
                    tmp_df = pd.concat([remain_df, tmp_df], axis = 1)
                
                tmp_df = tmp_df[sorted(df.columns)]

                missing, extreme, total = statistics.quality_issue(tmp_df)
                tmp_df_problem = pd.DataFrame({x for x in zip(column_list, total)}, columns = ['index', 'total'])
                tmp_df_problem = tmp_df_problem.sort_values(by = 'index').reset_index(drop = True)
                tmp_df_problem = tmp_df_problem.drop(['index'], axis = 1)
                
                tmp_df_problem = tmp_df_problem.sum().tolist()

                diff = current_df_problem[0] - tmp_df_problem[0]
                output.append([i, diff])

            output.sort(key = lambda x:-x[1])

        except:
            flash("stop the profiling process")
            return redirect('/profiling_vis')
    
    if recommend_method == 'B':
        column_list = list(df)
        check_list = []

        for i in range(len(column_list)):
            check_list.append(df.iloc[:, i].isnull().values.any())

        tmp_df = df
        tmp_column_list = column_list

        for i in range(len(check_list)):
            if check_list[i] == False:
                column_name = column_list[i]

                tmp_df.drop([column_name], axis = 1)
                tmp_column_list.remove(column_name)

        try:
            # quality per column in current df
            output = statistics.quality_metric_total(tmp_df)
            df_problem = pd.DataFrame({x for x in zip(tmp_column_list, output[0], output[1], output[2])}, columns = ['index', 'kstest', 'skewness', 'kurtosis'])
            df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)

            df_problem['skewness'] = df_problem['skewness'].abs()
            df_problem['kurtosis'] = df_problem['kurtosis'].abs()

            # select the column with the lowest quality
            target_column = df_problem.iloc[df_problem[recommend_type].idxmax()]['index']
            target_idx = df.columns.get_loc(target_column)
            target_df = df.loc[:, [target_column]]
            remain_df = df.drop([target_column], axis = 1)

            # action
            # if target_column have missing value
            action = ['missing mean', 'missing median', 'missing em', 'missing locf', 'outlier mean', 'outlier median', 'outlier em', 'outlier locf', 'missing remove', 'outlier remove']
            action_df = [imputation.custom_imp_mean(target_column, target_df), imputation.custom_imp_median(target_column, target_df),
                        imputation.custom_imp_em(target_column, target_df), imputation.custom_imp_locf(target_column, target_df)]

            lower, upper = statistics.lower_upper(target_df, target_column)
            outlier_df = target_df[(target_df[target_column] < lower) | (target_df[target_column] > upper)]
            outlier_index = list(outlier_df.index)

            for i in range(0, 4):
                for j in outlier_index:
                    target_df.loc[j, target_column] = action_df[i].iloc[j][target_column]
                action_df.append(target_df)

            action_df.append('missing remove')
            action_df.append('outlier remove')
        
            # current column quality
            kstest, skewness, kurtosis = statistics.quality_metric(df, target_idx)
            if recommend_type == 'kstest': current_df_quality = kstest
            elif recommend_type == 'skewness': current_df_quality = abs(skewness)
            elif recommend_type == 'kurtosis': current_df_quality = abs(kurtosis)

            # compare current - action
            output = []

            for i in range(0, 10):
                if action[i] == 'missing remove':
                    tmp_df = df.dropna(subset = [target_column], how = 'all')
                elif action[i] == 'outlier remove':
                    tmp_df = df.drop(outlier_df.index, axis = 0).reset_index(drop = True)
                else:
                    tmp_df = action_df[i]
                    tmp_df = pd.concat([remain_df, tmp_df], axis = 1)

                tmp_df = tmp_df[sorted(df.columns)]

                kstest, skewness, kurtosis = statistics.quality_metric(tmp_df, 0)
                if recommend_type == 'kstest': tmp_df_quality = kstest
                elif recommend_type == 'skewness': tmp_df_quality = abs(skewness)
                elif recommend_type == 'kurtosis': tmp_df_quality = abs(kurtosis)

                diff = current_df_quality - tmp_df_quality
                output.append([i, diff])
            
            output.sort(key = lambda x:-x[1])

        except:
            flash("stop the profiling process")
            return redirect('/profiling_vis')

    if recommend_method == 'C':
        column_list = list(df)

        ##### example
        # target column - quality issue
        missing, extreme, total = statistics.quality_issue(df)
        df_problem = pd.DataFrame({x for x in zip(column_list, missing, extreme, total)}, columns = ['index', 'missing', 'extreme', 'total'])
        df_problem = df_problem.sort_values(by = 'index').reset_index(drop = True)

        target_column = df_problem.iloc[df_problem['total'].idxmax()]['index']
        target_idx = df.columns.get_loc(target_column)
        target_df = df.loc[:, [target_column]]
        remain_df = df.drop([target_column], axis = 1)

        try:
            # feature
            feature = []

            # feature - base
            missing, outlier, incons = statistics.quality_issue(target_df)
            kstest, skewness, kurtosis = statistics.quality_metric(df, target_idx)
            base_feature = [missing[0], outlier[0], incons[0], kstest, skewness, kurtosis]

            # feature - rank
            rank_feature = []

            missing_list = []
            outlier_list = []
            incons_list = []
            kstest_list = []
            skewness_list = []
            kurtosis_list = []

            for i in range(len(column_list)):
                missing, outlier, incons = statistics.quality_issue(df.iloc[:, [i]])
                kstest, skewness, kurtosis = statistics.quality_metric(df, i)

                missing_list.append(missing)
                outlier_list.append(outlier)
                incons_list.append(incons)
                kstest_list.append(kstest)
                skewness_list.append(skewness)
                kurtosis_list.append(kurtosis)

            my_list = [missing_list[target_idx], outlier_list[target_idx], incons_list[target_idx],
                    kstest_list[target_idx], skewness_list[target_idx], kurtosis_list[target_idx]]

            missing_list.sort(reverse = True)
            outlier_list.sort(reverse = True)
            incons_list.sort(reverse = True)
            kstest_list.sort(reverse = True)
            skewness_list.sort(reverse = True)
            kurtosis_list.sort(reverse = True)

            rank_feature.append((missing_list.index(my_list[0]) + 1)/len(missing_list))
            rank_feature.append((outlier_list.index(my_list[1]) + 1)/len(outlier_list))
            rank_feature.append((incons_list.index(my_list[2]) + 1)/len(incons_list))
            rank_feature.append((kstest_list.index(my_list[3]) + 1)/len(kstest_list))
            rank_feature.append((skewness_list.index(my_list[4]) + 1)/len(skewness_list))
            rank_feature.append((kurtosis_list.index(my_list[5]) + 1)/len(kurtosis_list))

            tmp = base_feature + rank_feature
            feature.append(tmp)

            feature_df = pd.DataFrame(feature, columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
            feature_df = feature_df.append(feature_df, ignore_index = True)
            feature_df = feature_df.append(feature_df, ignore_index = True)
            feature_df = feature_df.drop([feature_df.index[3]])

            # tsne
            from sklearn.manifold import TSNE
            tsne = TSNE(random_state = 0)
            tsne_components = tsne.fit_transform(feature_df.to_numpy())
            tsne_df = pd.DataFrame(data = tsne_components, columns = ['component1', 'component2'])

            model = joblib.load('model.pkl')
            predict_action = model.predict(tsne_df)
            predict_action = predict_action.tolist()

            ##### example
            # action - remove, mean, em
            predict_action = [0, 1, 2]

            ##### example
            # value - missing
            action = []
            action_df = []

            for i in range(len(predict_action)):
                if predict_action[i] == 0:
                    action.append('missing remove')
                    action_df.append('missing remove')
                elif predict_action[i] == 1:
                    action.append('missing mean')
                    action_df.append(imputation.custom_imp_mean(target_column, target_df))
                elif predict_action[i] == 2:
                    action.append('missing em')
                    action_df.append(imputation.custom_imp_em(target_column, target_df))                
                elif predict_action[i] == 3:
                    action.append('missing locf')
                    action_df.append(imputation.custom_imp_locf(target_column, target_df))                
                elif predict_action[i] == 4:
                    action.append('missing median')
                    action_df.append(imputation.custom_imp_median(target_column, target_df))

            output = [[predict_action[0], 0], [predict_action[1], 0], [predict_action[2], 0]]

        except:
            flash("stop the profiling process")
            return redirect('/profiling_vis')            

    # select the 3 actions
    recommend_idx = [output[0][0], output[1][0], output[2][0]]

    # recommend - children node
    for i in range(0, 3):
        if action[recommend_idx[i]] == 'missing remove':
            recommend_df = df.dropna(subset = [target_column], how = 'all')
        elif action[recommend_idx[i]] == 'outlier remove':
            recommend_df = df.drop(outlier_df.index, axis = 0).reset_index(drop = True)
        else:
            recommend_df = action_df[recommend_idx[i]]
            recommend_df = pd.concat([remain_df, recommend_df], axis = 1)
        
        recommend_df = recommend_df[sorted(df.columns)]

        global cnt
        recommend_df.to_csv('static/data/' + str(cnt) + '.csv', index = False)

        # generation children node
        new_node = tree.TreeNode(file = str(cnt), name = str(cnt), state = '', action = action[recommend_idx[i]])
        root.add_child_to(name, new_node)

        cnt = cnt + 1

    # update state
    root.update_state(name)

    output_data = root.tree_to_dict()
    with open('static/data/tree_data.json', 'w') as f:
        json.dump(output_data, f, indent = 4)

    return redirect('/profiling_vis')

if __name__ == '__main__':
  app.jinja_env.auto_reload = True
  app.config['TEMPLATES_AUTO_RELOAD'] = True
  app.run(debug = True)