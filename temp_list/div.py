def input_query_div(cnt_create, attributes, tasks, visType):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 250px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="list" style="width: 20%;">
                <table>
                    <tr>
                        <th>attributes</th>
                        <td>''' + attributes + '''</td>
                    </tr>
                    <tr>
                        <th>tasks</th>
                        <td>''' + tasks + '''</td>
                    </tr>
                    <tr>
                        <th>visType</th>
                        <td>''' + visType + '''</td>
                    </tr>
                </table>
            </div>
            <div id="vlSpec" style="width: 30%;"></div>
        </div>
        '''
    return div

def current_div_A(cnt_create, name, children_node, action, output_value, output_percent, output_sign):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 250px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="node" style="width: 30px; height: 30px; margin: 5px; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + name + '''</div>
            <div id="bar1" style="width: 23%;"></div>
            <div id="bar2" style="width: 23%;"></div>
            <div id="recommend" style="width: 50%; margin-top: 30px;">
                <div id="recommend1" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node1" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + children_node[0].name + '''</div>
                    <div id="recommend_list1" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[0] + '''
                        <div id="issue" style="display: flex;">
                            missing value
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[0][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[0][0]) + '''</div>
                            </div>
                            '''+ output_sign[0][0] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[0][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[0][1]) + '''</div>
                            </div>
                            '''+ output_sign[0][1] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[0][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[0][2]) + '''</div>
                            </div>
                            '''+ output_sign[0][2] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend2" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node2" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[1].name + '''</div>
                    <div id="recommend_list2" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[1] + '''
                        <div id="issue" style="display: flex;">
                            missing value
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[1][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[1][0]) + '''</div>
                            </div>
                            '''+ output_sign[1][0] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[1][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[1][1]) + '''</div>
                            </div>
                            '''+ output_sign[1][1] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[1][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[1][2]) + '''</div>
                            </div>
                            '''+ output_sign[1][2] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend3" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node3" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[2].name + '''</div>
                    <div id="recommend_list3" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[2] + '''
                        <div id="issue" style="display: flex;">
                            missing value
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[2][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[2][0]) + '''</div>
                            </div>
                            '''+ output_sign[2][0] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[2][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[2][1]) + '''</div>
                            </div>
                            '''+ output_sign[2][1] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[2][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[2][2]) + '''</div>
                            </div>
                            '''+ output_sign[2][2] + '''
                        </div>
                    </div>
                </div>
            </div>
        </div>
    '''
    return div

def current_div_B(cnt_create, name, children_node, action, output_value, output_percent, output_sign):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 250px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="node" style="width: 30px; height: 30px; margin: 5px; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + name + '''</div>
            <div id="bar1" style="width: 27%;"></div>
            <div id="bar2" style="width: 27%;"></div>
            <div id="recommend" style="width: 45%; margin-top: 30px;">
                <div id="recommend1" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node1" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + children_node[0].name + '''</div>
                    <div id="recommend_list1" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[0] + '''
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[0][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[0][0]) + '''</div>
                            </div>
                            '''+ output_sign[0][0] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[0][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[0][1]) + '''</div>
                            </div>
                            '''+ output_sign[0][1] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[0][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[0][2]) + '''</div>
                            </div>
                            '''+ output_sign[0][2] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend2" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node2" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[1].name + '''</div>
                    <div id="recommend_list2" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[1] + '''
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[1][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[1][0]) + '''</div>
                            </div>
                            '''+ output_sign[1][0] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[1][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[1][1]) + '''</div>
                            </div>
                            '''+ output_sign[1][1] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[1][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[1][2]) + '''</div>
                            </div>
                            '''+ output_sign[1][2] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend3" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node3" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[2].name + '''</div>
                    <div id="recommend_list3" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[2] + '''
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[2][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[2][0]) + '''</div>
                            </div>
                            '''+ output_sign[2][0] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[2][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[2][1]) + '''</div>
                            </div>
                            '''+ output_sign[2][1] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[2][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[2][2]) + '''</div>
                            </div>
                            '''+ output_sign[2][2] + '''
                        </div>
                    </div>
                </div>
            </div>
        </div>
    '''
    return div    


def current_div_C(cnt_create, name, children_node, action, output_value, output_percent, output_sign):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 380px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="node" style="width: 30px; height: 30px; margin: 5px; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + name + '''</div>
            <div id="bar1" style="width: 27%;"></div>
            <div id="bar2" style="width: 27%;"></div>
            <div id="recommend" style="width: 45%; margin-top: 30px;">
                <div id="recommend1" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node1" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + children_node[0].name + '''</div>
                    <div id="recommend_list1" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[0] + '''
                        <div id="issue" style="display: flex;">
                            missing
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[0][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[0][0]) + '''</div>
                            </div>
                            '''+ output_sign[0][0] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[1][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[1][0]) + '''</div>
                            </div>
                            '''+ output_sign[1][0] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[2][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[2][0]) + '''</div>
                            </div>
                            '''+ output_sign[2][0] + '''
                        </div>
                        <div style="display: block;"></div>
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[3][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[3][0]) + '''</div>
                            </div>
                            '''+ output_sign[3][0] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[4][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[4][0]) + '''</div>
                            </div>
                            '''+ output_sign[4][0] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[5][0]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[5][0]) + '''</div>
                            </div>
                            '''+ output_sign[5][0] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend2" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node2" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[1].name + '''</div>
                    <div id="recommend_list2" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[1] + '''
                        <div id="issue" style="display: flex;">
                            missing
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[0][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[0][1]) + '''</div>
                            </div>
                            '''+ output_sign[0][1] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[1][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[1][1]) + '''</div>
                            </div>
                            '''+ output_sign[1][1] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[2][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[2][1]) + '''</div>
                            </div>
                            '''+ output_sign[2][1] + '''
                        </div>
                        <div style="display: block;"></div>
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[3][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[3][1]) + '''</div>
                            </div>
                            '''+ output_sign[3][1] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[4][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[4][1]) + '''</div>
                            </div>
                            '''+ output_sign[4][1] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[5][1]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[5][1]) + '''</div>
                            </div>
                            '''+ output_sign[5][1] + '''
                        </div>
                    </div>
                </div>
                <br>
                <div id="recommend3" style="display: flex; width: 100%; height: 20%;">
                    <div id="recommend_node3" style="width: 30px; height: 30px; margin: 5px; background-color: #e0e0e0; text-align: center; border: 2px solid; border-radius: 50%; line-height: 30px;">''' + children_node[2].name + '''</div>
                    <div id="recommend_list3" style="width: 100%; height: 100%; margin: 5px; background-color: #e0e0e0; text-align: left; border: 2px solid;">
                        ''' + action[2] + '''
                        <div id="issue" style="display: flex;">
                            missing
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[0][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[0][2]) + '''</div>
                            </div>
                            '''+ output_sign[0][2] + ''', outlier
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[1][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[1][2]) + '''</div>
                            </div>
                            '''+ output_sign[1][2] + ''', total
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[2][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #1f78b4 0%, white 100%);">''' + str(output_value[2][2]) + '''</div>
                            </div>
                            '''+ output_sign[2][2] + '''
                        </div>
                        <div style="display: block;"></div>
                        <div id="issue" style="display: flex;">
                            kstest
                            <div id="issue1" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue1_inner" style="width: ''' + str(output_percent[3][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #6a3d9a 0%, white 100%);">''' + str(output_value[3][2]) + '''</div>
                            </div>
                            '''+ output_sign[3][2] + ''', skew
                            <div id="issue2" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue2_inner" style="width: ''' + str(output_percent[4][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #ff7f00 0%, white 100%);">''' + str(output_value[4][2]) + '''</div>
                            </div>
                            '''+ output_sign[4][2] + ''', kurto
                            <div id="issue3" style="width: 10%; height: 20px; background-color: white; border: 2px solid;">
                                <div id="issue3_inner" style="width: ''' + str(output_percent[5][2]) + '''%; height: 20px; background-size: 100%''' + ''' 100%; text-align: center; background-image: linear-gradient(90deg, #33a02c 0%, white 100%);">''' + str(output_value[5][2]) + '''</div>
                            </div>
                            '''+ output_sign[5][2] + '''
                        </div>
                    </div>
                </div>
            </div>
        </div>
    '''
    return div


def recommend_div_A(cnt_create, name):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 250px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="node" style="width: 30px; height: 30px; margin: 5px; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + name + '''</div>
            <div id="heatmap_before" style="width: 23%;"></div>
            <div id="heatmap_after" style="width: 23%;"></div>
            <div id="histogram_before" style="width: 23%;"></div>
            <div id="histogram_after" style="width: 23%;"></div>
        </div>
    '''
    return div

def recommend_div_B(cnt_create, name):
    div = '''
        <div id="vis''' + str(cnt_create) + '''" style="display: flex; height: 250px; margin-bottom: 5px; border: 3px solid #9e9e9e; font-size: 17px;">
            <div id="node" style="width: 30px; height: 30px; margin: 5px; text-align: center; border: 2px solid black; border-radius: 50%; line-height: 30px;">''' + name + '''</div>
            <div id="ecdf_before" style="width: 23%;"></div>
            <div id="ecdf_after" style="width: 23%;"></div>
            <div id="density_before" style="width: 23%;"></div>
            <div id="density_after" style="width: 23%;"></div>
        </div>

    '''
    return div