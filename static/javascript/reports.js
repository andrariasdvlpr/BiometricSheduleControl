function onLoadReports(){
    google.charts.load("current", {'packages':["corechart"]});
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){
                var histograms = JSON.parse(this.responseText);  
                google.charts.setOnLoadCallback(drawCharthistogram(histograms));
                google.charts.setOnLoadCallback(drawChart(histograms));
            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.18/bsc/rest-histograms-data/";
    xmlhttp.open("GET", url, true);
    xmlhttp.send();

    function drawCharthistogram(object) { 

        var diff = new google.visualization.DataTable();
        diff.addColumn('number','distance');
        diff.addRows(object.length);
        for(i=0;i<object.length;i++){
            diff.setValue(i,0,object[i].diff_distance);
        }
        var options_diff = {
            title: 'Intentos verificación impostores',
            legend: { position: 'none' },
            colors: ['#F14A29'],
            'width': 500,
            'height': 400,
            'chartArea': {'width': '80%', 'height': '80%'},

        };

        var chart = new google.visualization.Histogram(document.getElementById('chart_div_2'));
        chart.draw(diff, options_diff);
    }

    function drawChart(object) {
        var same = new google.visualization.DataTable();
        same.addColumn('number','distance');   
        same.addRows(object.length);
        for(i=0;i<object.length;i++){
            same.setValue(i,0,object[i].same_distance);
        }

        var options = {
            title: 'Intentos verificación usuarios',
            legend: { position: 'none' },
            colors: ['#5ce718'],
            'width': 500,
            'height': 400,
            'chartArea': {'width': '80%', 'height': '80%'},
        };
    
        var chart = new google.visualization.Histogram(document.getElementById('chart_div'));
        chart.draw(same, options);
    }

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){
                var roc_data = JSON.parse(this.responseText);  
                google.charts.setOnLoadCallback(drawChartRoc(roc_data));
            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.18/bsc/rest-roc-data/";
    xmlhttp.open("GET", url, true);
    xmlhttp.send();

    function drawChartRoc(object) {

        var data = new google.visualization.DataTable();
        data.addColumn('number', 'FPR');
        data.addColumn('number', 'ROC');
        data.addColumn('number', 'Line');
        data.addRows(object.length);
        for(i=0;i<object.length;i++){
            data.setValue(i,0,object[i].FPR);
            data.setValue(i,1,object[i].ROC);
            data.setValue(i,2,object[i].Line);
        }

        var options = {
        width: 500,
        height: 400,
        chartArea: {'width': '65%', 'height': '80%'},

        };

        var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));
        chart.draw(data, options);
    }

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){
                var det_data = JSON.parse(this.responseText);  
                google.charts.setOnLoadCallback(drawChartDET(det_data));
            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.18/bsc/rest-det-data/";
    xmlhttp.open("GET", url, true);
    xmlhttp.send();

    function drawChartDET(object) {

        var data = new google.visualization.DataTable();
        data.addColumn('number', 'FPR');
        data.addColumn('number', 'DET');
        data.addRows(object.length);
        for(i=0;i<object.length;i++){
            data.setValue(i,0,object[i].FPR_log);
            data.setValue(i,1,object[i].DET_log);
        }

        var options = {
        width: 500,
        height: 400,
        chartArea: {'width': '65%', 'height': '80%'},

        };

        var chart = new google.visualization.LineChart(document.getElementById('curve_chart2'));
        chart.draw(data, options);
    
    }
}