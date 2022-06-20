import pandas as pd
import numpy as np
from scipy import stats

def make_heatmap_df(column_list, current_df, df):
    df = df[sorted(df.columns)]
    size = int(len(current_df)/10)
    
    tmp_index, tmp_y, tmp_value = [], [], []
    data = {}

    for i in range(len(column_list)):
        tmp_df = df[column_list[i]]

        for j in range(int(len(current_df)/size)):
            slice_df = tmp_df.iloc[size * j : size * (j + 1)]
            missing = slice_df.isnull().sum()

            tmp_index.append(column_list[i])
            tmp_y.append(size * (j + 1))
            tmp_value.append(missing)

    data['index'] = tmp_index
    data['y'] = tmp_y
    data['value'] = tmp_value
    
    output_df = pd.DataFrame(data)
    return output_df, tmp_y

def make_density_df(df1, df2, state1, state2):
    tmp_index = []
    tmp_value  = []
    data = {}

    for i in range(len(df1)):
        tmp_index.append(state1)
        tmp_value.append(df1.iloc[i][0])

    for i in range(len(df2)):
        tmp_index.append(state2)
        tmp_value.append(df2.iloc[i][0])
    
    data['index'] = tmp_index
    data['value'] = tmp_value

    output_df = pd.DataFrame(data)

    return output_df

def make_ecdf_df(df, index):
    tmp_index, tmp_x, tmp_y = [], [], []
    data = {}

    if type(df) == 'pandas.core.frame.DataFrame':
        x = np.sort(df.to_numpy())
    else:
        x = np.sort(df)
        
    y = 1. * np.arange(len(x))/float(len(x) - 1)

    for i in range(0,len(x)):
        tmp_index.append(index)
        tmp_x.append(x[i])
        tmp_y.append(y[i])

    data['index'] = tmp_index
    data['x'] = tmp_x
    data['y'] = tmp_y

    output_df = pd.DataFrame(data)
    return output_df  

def svg_vlSpec(cnt_create, vlSpec):
    svg = '''
        var vlSpec = ''' + str(vlSpec) + ''';
        vegaEmbed('#vis''' + str(cnt_create) + ''' #vlSpec', vlSpec, {"actions": false});
    '''
    return svg

def svg_bar1(cnt_create, max_value):
    svg = '''
        var margin_bar = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            bar_width = 500 - margin_bar.left - margin_bar.right,
            bar_height = 280 - margin_bar.top - margin_bar.bottom;

        var svg_bar1_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #bar1")
            .append("svg")
            .attr("width", bar_width + margin_bar.left + margin_bar.right)
            .attr("height", bar_height + margin_bar.top + margin_bar.bottom)
            .append("g")
            .attr("transform","translate(" + margin_bar.left + "," + margin_bar.top + ")");

        d3v4.csv("static/data/bar_''' + str(cnt_create) + '''.csv", function(data) {
            var bar_subgroups = data.columns.slice(1)
            var bar_groups = d3v4.map(data, function(d){return(d.index)}).keys()

            var x1 = d3v4.scaleBand()
                .domain(bar_groups)
                .range([0, bar_width - 150])
                .padding([0.2])
            svg_bar1_''' + str(cnt_create) + '''.append("g")
                .attr("transform", "translate(0," + bar_height + ")")
                .call(d3v4.axisBottom(x1).tickSizeOuter(0));

            var y1 = d3v4.scaleLinear()
                .domain([0, ''' + str(max_value) + '''])
                .range([bar_height, 20]);
            svg_bar1_''' + str(cnt_create) + '''.append("g")
                .call(d3v4.axisLeft(y1));

            var xSubgroup = d3v4.scaleBand()
                .domain(bar_subgroups)
                .range([0, x1.bandwidth()])
                .padding([0.05])

            var color = d3v4.scaleOrdinal()
                .domain(bar_subgroups)
                .range(['#ff7f00','#33a02c'])

            svg_bar1_''' + str(cnt_create) + '''.append("g")
                .selectAll("g")
                .data(data)
                .enter()
                .append("g")
                .attr("transform", function(d) { return "translate(" + x1(d.index) + ",0)"; })
                .selectAll("rect")
                .data(function(d) { return bar_subgroups.map(function(key) { return {key: key, value: d[key]}; }); })
                .enter().append("rect")
                .attr("x", function(d) { return xSubgroup(d.key); })
                .attr("y", function(d) { return y1(d.value); })
                .attr("width", xSubgroup.bandwidth())
                .attr("height", function(d) { return bar_height - y1(d.value); })
                .attr("fill", function(d) { return color(d.key); });
            })

        svg_bar1_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).style("fill", "#ff7f00")
        svg_bar1_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("missing").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_bar1_''' + str(cnt_create) + '''.append("circle").attr("cx", 100).attr("cy", 0).attr("r", 6).style("fill", "#33a02c")
        svg_bar1_''' + str(cnt_create) + '''.append("text").attr("x", 110).attr("y", 0).text("outlier").style("font-size", "15px").attr("alignment-baseline","middle")
    '''
    return svg

def svg_bar2(cnt_create, max_value):
    svg = '''
        var margin_bar = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            bar_width = 500 - margin_bar.left - margin_bar.right,
            bar_height = 280 - margin_bar.top - margin_bar.bottom;

        var svg_bar2_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #bar2")
            .append("svg")
            .attr("width", bar_width + margin_bar.left + margin_bar.right)
            .attr("height", bar_height + margin_bar.top + margin_bar.bottom)
            .append("g")
            .attr("transform","translate(" + margin_bar.left + "," + margin_bar.top + ")");

        d3v4.csv("static/data/bar_''' + str(cnt_create) + '''.csv", function(data) {
            var bar_subgroups = data.columns.slice(1)
            var bar_groups = d3v4.map(data, function(d){return(d.index)}).keys()

            var x2 = d3v4.scaleBand()
                .domain(bar_groups)
                .range([0, bar_width - 150])
                .padding([0.5])
            svg_bar2_''' + str(cnt_create) + '''.append("g")
                .attr("transform", "translate(0," + bar_height + ")")
                .call(d3v4.axisBottom(x2).tickSizeOuter(0));

            var y2 = d3v4.scaleLinear()
                .domain([0, ''' + str(max_value) + '''])
                .range([bar_height, 20]);
            svg_bar2_''' + str(cnt_create) + '''.append("g")
                .call(d3v4.axisLeft(y2));

            var color = d3v4.scaleOrdinal()
                .domain(bar_subgroups)
                .range(['#ff7f00','#33a02c'])

            var stackedData = d3v4.stack()
                .keys(bar_subgroups)
                (data)

            svg_bar2_''' + str(cnt_create) + '''.append("g")
                .selectAll("g")
                .data(stackedData)
                .enter().append("g")
                .attr("fill", function(d) { return color(d.key); })
                .selectAll("rect")

                .data(function(d) { return d; })
                .enter().append("rect")
                .attr("x", function(d) { return x2(d.data.index); })
                .attr("y", function(d) { return y2(d[1]); })
                .attr("height", function(d) { return y2(d[0]) - y2(d[1]); })
                .attr("width", x2.bandwidth())
            })

        svg_bar2_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).style("fill", "#ff7f00")
        svg_bar2_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("missing").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_bar2_''' + str(cnt_create) + '''.append("circle").attr("cx", 100).attr("cy", 0).attr("r", 6).style("fill", "#33a02c")
        svg_bar2_''' + str(cnt_create) + '''.append("text").attr("x", 110).attr("y", 0).text("outlier").style("font-size", "15px").attr("alignment-baseline","middle")
    '''
    return svg

def svg_bar3(cnt_create, max_value, div_type):
    svg = '''
        var margin_bar = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            bar_width = 500 - margin_bar.left - margin_bar.right,
            bar_height = 280 - margin_bar.top - margin_bar.bottom;

        var svg_bar3_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #''' + div_type + '''")
            .append("svg")
            .attr("width", bar_width + margin_bar.left + margin_bar.right)
            .attr("height", bar_height + margin_bar.top + margin_bar.bottom)
            .append("g")
            .attr("transform", "translate(" + margin_bar.left + "," + margin_bar.top + ")");

        d3v4.csv("static/data/bar1_''' + str(cnt_create) + '''.csv", function(data) {
            var bar_subgroups = data.columns.slice(1)
            var bar_groups = d3v4.map(data, function(d){return(d.index)}).keys()

            var x1 = d3v4.scaleBand()
                .domain(bar_groups)
                .range([0, bar_width - 150])
                .padding([0.2])
            svg_bar3_''' + str(cnt_create) + '''.append("g")
                .attr("transform", "translate(0," + bar_height + ")")
                .call(d3v4.axisBottom(x1).tickSizeOuter(0));

            var y1 = d3v4.scaleLinear()
                .domain([0, ''' + str(max_value) + '''])
                .range([bar_height, 20]);
            svg_bar3_''' + str(cnt_create) + '''.append("g")
                .call(d3v4.axisLeft(y1));

            var xSubgroup = d3v4.scaleBand()
                .domain(bar_subgroups)
                .range([0, x1.bandwidth()])
                .padding([0.05])

            var color = d3v4.scaleOrdinal()
                .domain(bar_subgroups)
                .range(['#6a3d9a','#ff7f00','#33a02c'])

            svg_bar3_''' + str(cnt_create) + '''.append("g")
                .selectAll("g")
                .data(data)
                .enter()
                .append("g")
                .attr("transform", function(d) { return "translate(" + x1(d.index) + ",0)"; })
                .selectAll("rect")
                .data(function(d) { return bar_subgroups.map(function(key) { return {key: key, value: d[key]}; }); })
                .enter().append("rect")
                .attr("x", function(d) { return xSubgroup(d.key); })
                .attr("y", function(d) { return y1(d.value); })
                .attr("width", xSubgroup.bandwidth())
                .attr("height", function(d) { return bar_height - y1(d.value); })
                .attr("fill", function(d) { return color(d.key); });
            })

        svg_bar3_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).style("fill", "#6a3d9a")
        svg_bar3_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("kstest").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_bar3_''' + str(cnt_create) + '''.append("circle").attr("cx", 100).attr("cy", 0).attr("r", 6).style("fill", "#ff7f00")
        svg_bar3_''' + str(cnt_create) + '''.append("text").attr("x", 110).attr("y", 0).text("skewness").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_bar3_''' + str(cnt_create) + '''.append("circle").attr("cx", 200).attr("cy", 0).attr("r", 6).style("fill", "#33a02c")
        svg_bar3_''' + str(cnt_create) + '''.append("text").attr("x", 210).attr("y", 0).text("kurtosis").style("font-size", "15px").attr("alignment-baseline","middle")
    '''
    return svg

def svg_bar4(cnt_create, max_value, recommend_type):
    if recommend_type == 'kstest': color = '#6a3d9a'
    elif recommend_type == 'skewness': color = '#ff7f00'
    elif recommend_type == 'kurtosis': color = '#33a02c'

    svg = '''
        var margin_bar = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            bar_width = 500 - margin_bar.left - margin_bar.right,
            bar_height = 280 - margin_bar.top - margin_bar.bottom;

        var svg_bar4_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #bar2")
            .append("svg")
            .attr("width", bar_width + margin_bar.left + margin_bar.right)
            .attr("height", bar_height + margin_bar.top + margin_bar.bottom)
            .append("g")
            .attr("transform", "translate(" + margin_bar.left + "," + margin_bar.top + ")");

        d3v4.csv("static/data/bar2_''' + str(cnt_create) + '''.csv", function(data) {
            var x = d3v4.scaleBand()
                .range([ 0, bar_width - 150])
                .domain(data.map(function(d) { return d.index; }))
                .padding(0.5);
            svg_bar4_''' + str(cnt_create) + ''' .append("g")
                .attr("transform", "translate(0," + bar_height + ")")
                .call(d3v4.axisBottom(x).tickSizeOuter(0))
                .selectAll("text");

            var y = d3v4.scaleLinear()
                .domain([0, ''' + str(max_value) + '''])
                .range([ bar_height, 20]);
            svg_bar4_''' + str(cnt_create) + ''' .append("g")
                .call(d3v4.axisLeft(y));

            svg_bar4_''' + str(cnt_create) + ''' .selectAll("mybar")
                .data(data)
                .enter()
                .append("rect")
                .attr("x", function(d) { return x(d.index); })
                .attr("y", function(d) { return y(d.''' + recommend_type + '''); })
                .attr("width", x.bandwidth())
                .attr("height", function(d) { return bar_height - y(d.''' + recommend_type + '''); })
                .attr("fill", "''' + color + '''")
            })

        svg_bar4_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).style("fill", "''' + color + '''")
        svg_bar4_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("''' + recommend_type + '''").style("font-size", "15px").attr("alignment-baseline","middle")
    '''
    return svg

def svg_heatmap(cnt_create, column_list, y_list, target_column, state, div_name):
    svg = '''
        var margin_heatmap = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            heatmap_width = 530 - margin_heatmap.left - margin_heatmap.right,
            heatmap_height = 280 - margin_heatmap.top - margin_heatmap.bottom;

        var svg_heatmap''' + str(state) + '''_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #''' + str(div_name) + '''")
            .append("svg")
            .attr("width", heatmap_width + margin_heatmap.left + margin_heatmap.right)
            .attr("height", heatmap_height + margin_heatmap.top + margin_heatmap.bottom)
            .append("g")
            .attr("transform", "translate(" + margin_heatmap.left + "," + margin_heatmap.top + ")");

        var heatmap_groups = ''' + str(column_list) + '''
        var heatmap_y = ''' + str(y_list) + '''

        var x = d3v4.scaleBand()
            .range([ 0, heatmap_width - 150])
            .domain(heatmap_groups)
            .padding(0.01);
        svg_heatmap''' + str(state) + '''_''' + str(cnt_create) + '''.append("g")
            .attr("transform", "translate(0," + heatmap_height + ")")
            .call(d3v4.axisBottom(x))

        var y = d3v4.scaleBand()
            .range([ heatmap_height, 0 ])
            .domain(heatmap_y)
            .padding(0.01);
        svg_heatmap''' + str(state) + '''_''' + str(cnt_create) + '''.append("g")
            .call(d3v4.axisLeft(y));

        var color1 = d3v4.scaleLinear()
            .range(["white", "#9e9e9e"])
            .domain([1, 100])

        var color2 = d3v4.scaleLinear()
            .range(["white", "#ff7f00"])
            .domain([1, 100])

        d3v4.csv("static/data/heatmap''' + str(state) + '''_''' + str(cnt_create) + '''.csv", function(data) {
        svg_heatmap''' + str(state) + '''_''' + str(cnt_create) + '''.selectAll()
            .data(data, function(d) {return d.index + ':' + d.y;})
            .enter()
            .append("rect")
            .attr("x", function(d) { return x(d.index) })
            .attr("y", function(d) { return y(d.y) })
            .attr("width", x.bandwidth() )
            .attr("height", y.bandwidth() )
            .style("fill", function(d) {
                if (d.index == "''' + str(target_column) + '''") { return color2(d.value * 20); }
                else { return color1(d.value * 20); }
            })
        })
    '''
    return svg

def svg_histogram1(cnt_create, target_column, min_value, max_value, lower, upper, state, div_name):
    svg = '''
        var margin_histogram = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            histogram_width = 530 - margin_histogram.left - margin_histogram.right,
            histogram_height = 280 - margin_histogram.top - margin_histogram.bottom;

        var svg_histogram1_''' + str(state) + '''_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #''' + div_name + '''")
            .append("svg")
            .attr("width", histogram_width + margin_histogram.left + margin_histogram.right)
            .attr("height", histogram_height + margin_histogram.top + margin_histogram.bottom)
            .append("g")
            .attr("transform", "translate(" + margin_histogram.left + "," + margin_histogram.top + ")");

        d3v4.csv("static/data/histogram''' + str(state) + '''_''' + str(cnt_create) + '''.csv", function(data) {
        var x = d3v4.scaleLinear()
            .domain([''' + str(min_value - 1) + ''', ''' + str(max_value + 1) + '''])
            .range([0, histogram_width - 150]);
        svg_histogram1_''' + str(state) + '''_''' + str(cnt_create) + '''.append("g")
            .attr("transform", "translate(0," + histogram_height + ")")
            .call(d3v4.axisBottom(x));

        var histogram = d3v4.histogram()
            .value(function(d) { return d.''' + str(target_column) + '''; })
            .domain(x.domain())
            .thresholds(x.ticks(70));

        var bins = histogram(data);

        var y = d3v4.scaleLinear()
            .range([histogram_height, 0]);
            y.domain([0, d3v4.max(bins, function(d) { return d.length; })]);
        svg_histogram1_''' + str(state) + '''_''' + str(cnt_create) + '''.append("g")
            .call(d3v4.axisLeft(y));

        svg_histogram1_''' + str(state) + '''_''' + str(cnt_create) + '''.selectAll("rect")
            .data(bins)
            .enter()
            .append("rect")
            .attr("x", 1)
            .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; })
            .attr("width", function(d) { return x(d.x1) - x(d.x0) - 1 ; })
            .attr("height", function(d) { return histogram_height - y(d.length); })
            .style("fill", function(d) {
                if (d.x0 > ''' + str(upper) + ''') { return "#33a02c"; }
                else if (d.x0 < ''' + str(lower) + ''') { return "#33a02c"; }
                else { return "#9e9e9e"; }
            })
        });
    '''
    return svg

def svg_density(cnt_create, recommend_type, min_value, max_value, state_num, state_str, div_name):
    if recommend_type == 'skewness': color = '#ff7f00'
    elif recommend_type == 'kurtosis': color = '#33a02c'

    svg = '''
        var margin_density = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            density_width = 530 - margin_density.left - margin_density.right,
            density_height = 280 - margin_density.top - margin_density.bottom;

        var svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #''' + div_name + '''")
            .append("svg")
            .attr("width", density_width + margin_density.left + margin_density.right)
            .attr("height", density_height + margin_density.top + margin_density.bottom)
            .append("g")
            .attr("transform","translate(" + margin_density.left + "," + margin_density.top + ")");

        d3v4.csv("static/data/density''' + str(state_num) + '''_''' + str(cnt_create) + '''.csv", function(data) {
        var x = d3v4.scaleLinear()
            .domain([''' + str(min_value - 10) + ''', ''' + str(max_value + 10) + '''])
            .range([0, density_width - 150]);
        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("g")
            .attr("transform", "translate(0," + density_height + ")")
            .call(d3v4.axisBottom(x));

        var y = d3v4.scaleLinear()
            .domain([0, 0.05])
            .range([density_height, 20]);
        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("g")
            .call(d3v4.axisLeft(y));

        var kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(60))
        var density1 =  kde( data
            .filter( function(d){return d.index === "normal"} )
            .map(function(d){  return d.value; }) )
        var density2 =  kde( data
            .filter( function(d){return d.index === "''' + state_str + '''"} )
            .map(function(d){  return d.value; }) )

        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("path")
            .attr("class", "mypath")
            .datum(density1)
            .attr("fill", "none")
            .attr("stroke", "#9e9e9e")
            .attr("stroke-width", 1.5)
            .attr("stroke-linejoin", "round")
            .attr("d",  d3v4.line()
            .curve(d3v4.curveBasis)
            .x(function(d) { return x(d[0]); })
            .y(function(d) { return y(d[1]); })
            );

        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("path")
            .attr("class", "mypath")
            .datum(density2)
            .attr("fill", "none")
            .attr("stroke", "''' + str(color) + '''")
            .attr("stroke-width", 1.5)
            .attr("stroke-linejoin", "round")
            .attr("d",  d3v4.line()
            .curve(d3v4.curveBasis)
            .x(function(d) { return x(d[0]); })
            .y(function(d) { return y(d[1]); })
            );

        });

        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).style("fill", "#6e6e6e")
        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("normal").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("circle").attr("cx", 100).attr("cy", 0).attr("r", 6).style("fill", "''' + str(color) + '''")
        svg_density'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("text").attr("x", 110).attr("y", 0).text("''' + state_str + '''").style("font-size", "15px").attr("alignment-baseline","middle")

        function kernelDensityEstimator(kernel, X) {
            return function(V) {
                return X.map(function(x) {
                return [x, d3v4.mean(V, function(v) { return kernel(x - v); })];
                });
            };
        }
        function kernelEpanechnikov(k) {
            return function(v) {
                return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
            };
        }
    '''

    return svg

def svg_ecdf(cnt_create, state_num, state_str, div_name):
    color = '#6a3d9a'

    svg = '''
        var margin_ecdf = {''' + '''top: 50, right: 50, bottom: 50, left: 50},
            ecdf_width = 530 - margin_ecdf.left - margin_ecdf.right,
            ecdf_height = 280 - margin_ecdf.top - margin_ecdf.bottom;

        var svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + ''' = d3v4.select("#vis''' + str(cnt_create) + ''' #''' + div_name + '''")
            .append("svg")
            .attr("width", ecdf_width + margin_ecdf.left + margin_ecdf.right)
            .attr("height", ecdf_height + margin_ecdf.top + margin_ecdf.bottom)
            .append("g")
            .attr("transform","translate(" + margin_ecdf.left + "," + margin_ecdf.top + ")");

        d3v4.csv("static/data/ecdf''' + str(state_num) + '''_''' + str(cnt_create) + '''.csv", function(data) {
            var sumstat = d3v4.nest()
                .key(function(d) { return d.index;})
                .entries(data);

            var x = d3v4.scaleLinear()
                .domain([d3v4.min(data, function(d) { return +d.x; }), d3v4.max(data, function(d) { return +d.x; })])
                .range([ 0, ecdf_width - 150 ]);
            svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("g")
                .attr("transform", "translate(0," + ecdf_height + ")")
                .call(d3v4.axisBottom(x));

            var y = d3v4.scaleLinear()
                .domain([0, d3v4.max(data, function(d) { return +d.y; })])
                .range([ ecdf_height, 20 ]);
            svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("g")
                .call(d3v4.axisLeft(y));

            var res = sumstat.map(function(d){ return d.key })
            var color = d3v4.scaleOrdinal()
                .domain(res)
                .range(["#9e9e9e","''' + str(color) + '''"])

            svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.selectAll(".line")
                .data(sumstat)
                .enter()
                .append("path")
                .attr("fill", "none")
                .attr("stroke", function(d){ return color(d.key) })
                .attr("stroke-width", 1.5)
                .attr("d", function(d){
                    return d3v4.line()
                        .x(function(d) { return x(d.x); })
                        .y(function(d) { return y(+d.y); })
                        (d.values)
                    })
                })

        svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("circle").attr("cx", 0).attr("cy",0).attr("r", 6).style("fill", "#6e6e6e")
        svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("text").attr("x", 10).attr("y", 0).text("normal").style("font-size", "15px").attr("alignment-baseline","middle")
        svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("circle").attr("cx", 100).attr("cy", 0).attr("r", 6).style("fill", "''' + str(color) + '''")
        svg_ecdf'''  + str(state_num) + '''_''' + str(cnt_create) + '''.append("text").attr("x", 110).attr("y", 0).text("''' + str(state_str) + '''").style("font-size", "15px").attr("alignment-baseline","middle")
    '''

    return svg