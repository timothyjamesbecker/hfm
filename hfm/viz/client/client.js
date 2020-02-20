//globals that ajax call attach to
var samples = null,
    sample_names = null,
    track_names = null,
    genes = null,
    ref_seqs = null,
    last_layers = null,
    last_scroll = null,
    last_seq = null,
    last_axis = null;
    scroll_top = null,
    scroll_sm = null,
    xScale = null,
    xScale2 = null,
    xAxis = null,
    brush = null,
    rest_host  = 'http://localhost:8080';

function genes_like(gene_stub){
    var search = [];
    for(g in genes){
        if(genes[g].toUpperCase().includes(gene_stub.toUpperCase())){
            search.push(genes[g]);
        }
    }
    return search;
}
function standard_moments(view,n){
    var M = [],
        mns = [0,0,0,-3];
    for(var i=0; i<view.length; i++) {
        mns = [mean(view[i]),std(view[i]),skew(view[i]),kurt(view[i])];
        var mnts = [];
        for(var j=0;j<n.length;j++){ mnts.push(mns[n[j]-1]); }
        M.push(mnts);
    }
    return M;
}
function normalize_moments(M){ //variable number of moment normalization
    var N = [],
        E = [];
    for(var j=0; j<M[0].length;j++) {
        E.push({'min':Number.MAX_VALUE,'max':Number.MAX_VALUE*-1,'diff':0});
        for (var i = 0; i < M.length; i++) {
            if (M[i][j] < E[j]['min']) { E[j]['min'] = M[i][j]; }
            if (M[i][j] > E[j]['max']) { E[j]['max'] = M[i][j];}
        }
        E[j]['diff'] = E[j]['max']-E[j]['min'];
        if (E[j]['diff'] <= 0) { E[j]['diff'] = 1.0;}
    }
    for(var i=0; i<M.length; i++){
        var ns = [];
        for(var j=0;j<M[0].length;j++){ ns.push((M[i][j]-E[j]['min'])/E[j]['diff']); }
        N.push(ns);
    }
    return {'N':N,'E':E};
}
function mean(M){
    return M[4];
} //central m1
function std(M){
    if(M[5]>0.0 && M[0]>1){
        return Math.pow(M[5]/(M[0]-1),0.5);
    }else{
        return 0.0;
    }
}  //central m2
function skew(M){
    if(M[5]>0.0){
        return Math.pow(M[0],0.5)*M[6]/Math.pow(M[5],1.5);
    }else{
        return 0.0;
    }

} //central m3
function kurt(M){
    if(M[5]>0.0){
        return M[0]*Math.pow(M[7]/M[5],2)-3.0;
    }else{
        return -3.0;
    }
} //central m4
function cor_moments(X,Y){
    var cor = [];
    for(var d=0; d<X[0].length; d++) {
        var n = X.length,
            buff = [0, 0, 0, 0, 0];
        for (var i = 0; i < n; i++) {
            buff[0] += X[i][d];
            buff[1] += Y[i][d];
            buff[2] += X[i][d]*Y[i][d];
            buff[3] += X[i][d]**2;
            buff[4] += Y[i][d]**2;
        }
        cor.push((buff[2]*n-buff[0]*buff[1])/Math.pow((buff[3]*n-buff[0]**2)*(buff[4]*n-buff[1]**2),0.5));
    }
    return cor;
}

function clear_sm(sm){
    d3.select('#div_'+sm).text(''); //clear existing tracks
}

function sm_order(sms,sm){
    var _sms = [],
        pos = {},
        e = 0;
    for(var i in sms){
        if(sms[i]!='all sm(s)') {
            _sms.push(sms[i]);
            if (sms[i] == sm) { e = i; }
        }
    }
    for(var i in sms){ pos[sms[i]] = Math.abs(e-i); }
    _sms.sort(function(a,b){
       return pos[a]-pos[b];
    });
    return _sms;
}

function render_axis_components(view){
    var coord = view['coord'];
    var margin = 10;
    var width = document.getElementById('main_container').clientWidth-margin,
        height = 200;

    xScale = d3.scaleLinear()
        .domain([coord[1], coord[2]])
        .range([margin, width-2*margin])
    xScale2 = d3.scaleLinear()
        .domain([0,ref_seqs[coord[0]]])
        .range([margin,width-3*margin])
    xAxis = d3.axisBottom().scale(xScale);

    d3.select('#nav_container').text(''); //clear the axis--------
    //try some bruch code here-------------------------------------------------
    var brush  = d3.select('#nav_container')
        .append('svg')
        .attr("width",width)
        .attr('height',height/6)
        .append("g")
        .attr("class", "brush")
        .style('margin-left',margin.toString+'px')
        .attr("transform", "translate("+margin.toString()+","+margin.toString()+")")
        .call(d3.brushX()
            .extent([[0, 0], [width, height/6]])
            .on("end", brushed));

    if(last_scroll!=null) {
        brush.call(d3.brushX().move, last_scroll);
    }else{
        var x1 = xScale2(coord[1])
        var x2 = xScale2(coord[2])
        if(x2-x1<2){
            x1 = Math.max(0,x1-1);
            x2 = Math.min(ref_seqs[coord[0]],x2+1);
        }
        last_scroll = [xScale2(coord[1]),xScale2(coord[2])];
        brush.call(d3.brushX().move,last_scroll);
    }
    //-------------------------------------------------------------------------
    function brushed() {
        if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
        var s = d3.event.selection;
        var new_coord = [coord[0],Math.round(xScale2.invert(s[0]+margin)),Math.round(xScale2.invert(s[1]))];
        last_scroll = s;
        last_axis = {'coord':new_coord,'arrow':true};
        $("#geneorpos_input").val(new_coord[0].toString()+":"+new_coord[1].toString()+"-"+new_coord[2].toString());
        render(d3.event.sourceEvent);
    }

    d3.select('body')
        .on("keydown", function() {
            if(d3.event.key == 'ArrowLeft'){
                console.log('left arrow keydown');
                var coord = last_axis['coord'];       //seq,start,end coordinates
                var m     = last_axis['axis']['length']; //number of windows
                var w     = last_axis['axis']['w'];      //window size
                var diff  = w*Math.round(m*0.1)     //how much to shift by...
                var new_coord = [coord[0],coord[1]-diff,coord[2]-diff];
                if(new_coord[1]>=0) {
                    last_axis['coord'] = new_coord;
                    last_axis['arrow'] = true;
                    $("#geneorpos_input").val(new_coord[0].toString() + ":" + new_coord[1].toString() + "-" + new_coord[2].toString());
                    render(d3.event.sourceEvent);
                }
            }
            if(d3.event.key == 'ArrowRight'){
                console.log('right arrow keydown');
                var coord = last_axis['coord'];       //seq,start,end coordinates
                var m     = last_axis['axis']['length']; //number of windows
                var w     = last_axis['axis']['w'];      //window size
                var diff  = w*Math.round(m*0.1)     //how much to shift by...
                var new_coord = [coord[0],coord[1]+diff,coord[2]+diff];
                if(new_coord[2]<=ref_seqs[coord[0]]) {
                    last_axis['coord'] = new_coord;
                    last_axis['arrow'] = true;
                    $("#geneorpos_input").val(new_coord[0].toString() + ":" + new_coord[1].toString() + "-" + new_coord[2].toString());
                    render(d3.event.sourceEvent);
                }
            }
        });

    d3.select('#nav_container')
        .append("svg")
        .attr("width", width)
        .attr("height", height/6)
        .append("g")
        .attr("transform", "translate(0," + margin.toString() + ")")
        .call(xAxis).select('path').style('opacity','0.1');
}

function render_tracks(view,sm){ //main method for new gene/pos query will refresh the page
    var coord = view['coord'],
        tt = 0;
    clear_sm(sm);
    tt = Object.keys(view[sm]).length;

    var margin = 10;
    var width = document.getElementById('main_container').clientWidth-margin,
        height = 200;

    //will need to do this again for a drag left or drag right
    //x-axis is added to the nav_conatiner once--------------------------------
    var cScale = d3.scaleLinear()
        .domain([0,tt-1])
        .range([0,275]);

    //now we work through each sample and each track
    var hue = -1;
    //console.log('processing sample: '+sm);
    for(var trk in view[sm]) {
        hue += 1; //start at 0
        var moments = standard_moments(view[sm][trk], [1, 2, 4]);
        var normalized = normalize_moments(moments);
        var normal = normalized['N'];
        var extrema = normalized['E'];

        //add new sample and track divs=============================
        d3.select('#div_'+sm).append('div')
            .attr('id','div_'+sm+'_'+trk);
        var svg = d3.select('#div_'+sm+'_'+trk).append('svg')
            .attr('width',width)
            .attr('height',height)
            .attr('id','svg_'+sm+'_'+trk)
            .on('mouseover',function(d,i){
                try {
                    var node_id = d3.select(this).attr('id').split('svg_')[1].split('_')[0];
                }catch{
                    console.log({'err':d3.select(this)});
                }
                scroll_sm = node_id; //save the last hovered sample name
            });
        //==========================================================

        var n = 3, // number of moments
            m = view[sm][trk].length, // number of samples per layer
            stack = d3.stack().keys(d3.range(n).map(function (d) {
                return "layer" + d;
            })).offset(d3.stackOffsetNone);

        //load up the stream layer data------------------
        var matrix = d3.range(m).map(function (d) {
            return {
                x: d,
                'layer0': normal[d][0],
                'layer1': normal[d][1],
                'layer2': normal[d][2],
                'layer3': normal[d][3]
            };
        });
        var layers = stack(matrix);
        var x = d3.scaleLinear()
            .domain([0, m-1])
            .range([margin, width-margin]);
        //y-axis for each track
        var y = d3.scaleLinear()
            .domain([0, n])
            .range([height, 0]);

        //coloring for each track using distributed hue red to violet----------------
        var color = ['hsla(' + cScale(hue).toString() + ',100%,50%,1.0)',
                     'hsla(' + cScale(hue).toString() + ',50%,50%,0.4)',
                     'hsla(' + cScale(hue).toString() + ',30%,50%,0.125)',
                     'hsla(' + cScale(hue).toString() + ',20%,50%,0.05)']

        //curved stacks generate the graph
        var area = d3.area()
            .curve(d3.curveCardinal)
            .x(function (d,i) { return x(d.data.x);})
            .y0(function (d) {  return y(d[0]); })
            .y1(function (d) {  return y(d[1]);});

        svg.selectAll(".layer")
            .data(layers)
            .enter().append("path")
            .attr('class','layer')
            .attr("d", area)
            .transition()
            .duration(100)
            .style("fill", function (d, i) {
                return color[i];
            });
        svg.append("rect")
            .attr("y", margin/2-4)
            .attr("width",width-margin)
            .attr("height",height)
            .attr("fill","none")
            .attr("stroke","grey")

        //y-axis on lower left up to the max mean value............................................................................................
        var yScale = d3.scaleLinear()
            .domain([extrema[0]['min'], extrema[0]['max']])
            .range([height / 3, margin])
        var yAxis = d3.axisRight().scale(yScale).tickValues([extrema[0]['min']+2, extrema[0]['min']+extrema[0]['diff']/2,extrema[0]['max']]);
        var yaxis_dom = svg.append("g")
            .attr("transform", "translate(0,"+(height-height/3).toString() + ")")
            .style('fill', color[0])
            .style('color', color[0])
            .style('stroke', color[0])
            .call(yAxis);
        yaxis_dom.select('path').style('stroke', color[0]).style('fill','rgb(60,60,60)');
        yaxis_dom.selectAll('.tick').select('line').style('stroke', color[0]).style('fill', color[0]);
        //..........................................................................................................................................

        //track text label in upper left hand corner-------------------------
        svg.append("g")
            .attr("transform","translate(0,"+height.toString()+")")
            .append("text")
            .style("fill",color[0])
            .style("font-size","200%")
            .style("stroke","black")
            .style("stroke-width","0.5px")
            .attr("x",3*margin)
            .attr("y",-2*height/3-margin)
            .text(sm+' : '+trk);
    }
}

function render(e){
    var flank = 0.25;
    last_scroll = null;
    scroll_top = $(window).scrollTop();
    if(e!=null){ e.preventDefault(); }
    if(last_axis!=null && 'arrow' in last_axis){ var flank = 0.0; } //no flank when arrow controls are used
    var tiles = Math.ceil(Number.parseFloat($('#tiles_input[name=tiles]').val().split('tiles:').join('')));
    var g = $('#coord_form input[name=gene]').val();
    if (g in ref_seqs) {//full sequence query
        url_stub = [rest_host,'/seq/'+g+'/start/0/end/'+ref_seqs[g]+'/tiles/'+tiles+'/flank/'+flank]
    } else {
        if (g.search(':') > 0 && g.search('-') > 0) {
            var parse = g.split(':');
            var seq = parse[0];
            parse = parse[1].split('-');
            var start = parse[0];
            var end = parse[1];
            url_stub = [rest_host,'/seq/'+seq+'/start/'+start+'/end/'+end+'/tiles/'+tiles+'/flank/'+flank];
        } else { //not a single chrom, not a chrom:start-end pattern
            url_stub = [rest_host,'/gene/'+g+'/tiles/'+tiles+'/flank/'+flank];
        }
    }
    var sms = $('#coord_form input[name=sms]').val().split(',');   //comma seperated
    var trks = $('#coord_form input[name=trks]').val().split(','); //comma seperated
    if (sms.length == 1 && sms[0] == 'all sm(s)') {
        sms = sample_names;
        var _sms = [];
        for(var i in sms){ if(sms[i]!='all sm(s)'){ _sms.push(sms[i]); }}
        sms = _sms;
    } //default to all sms
    if (trks.length == 1 && trks[0] == 'all trk(s)') {
        trks = track_names;
        var _trks = [];
        for(var i in trks){ if(trks[i]!='all trk(s)'){ _trks.push(trks[i]); }}
        trks = _trks;
    } //default to all trks
    //for loop to gather independantly each sample/track name needed for rendering
    if(scroll_sm!=null){
        sms_order = sm_order(sample_names,scroll_sm);
    }else{
        sms_order = [];
        for(var i in sample_names){
            if(sample_names[i]!='all sm(s)'){
                sms_order.push(sample_names[i]);
            }
        }
    }
    //get the overall axis using the first sms_order sm name
    var axis_url = url_stub[0]+'/sm/'+sms_order[0]+url_stub[1]+'/axis/1';
    $.ajax({ //async is synchronized so that all subsequent tracks can make use of the axis
        url:axis_url,
        data:{},
        success: function(result){
            if ('coord' in result && 'axis' in result) { //successfull REST response-------
                last_axis = {'coord':result['coord'],'axis':result['axis']};
                render_axis_components(last_axis); //get and cache the result
                for(var i in sms_order) {//each sample/track set:::::::::::::::::::::::::
                    if(sms.includes(sms_order[i])) {
                        var url = url_stub[0]+'/sm/'+sms_order[i]+url_stub[1]+'/axis/0';
                        get_tracks(url,sms_order[i],trks);
                    }else{ clear_sm(sms_order); }
                }//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            }
        }
    });
    $(window).scrollTop(scroll_top);
}

function get_tracks(url,sm,trks) {
    $.ajax({
        url: url,
        data: {},
        success: function (result) {
            if (result!=null && 'coord' in result) { //successfull REST response--------------------------------------------------------
                var query = {'coord': result['coord']};
                if (sm in result) {
                    if (!(sm in query)) { query[sm] = {}; }
                    for (var j in trks) {
                        if (trks[j] in result[sm]) { query[sm][trks[j]] = result[sm][trks[j]]; }
                    }
                }
                render_tracks(query,sm);
            }
        }
    });
}

$('#main_container').append('<div id="main_control_mask"></div>');
$('#main_control_mask').append('<div id="main_control_container"></div>');
$('#main_control_container').append('<form id="coord_form">hfm\n' +
    '            <span class="ui-widget"><input id="sms_input" type="text" name="sms" class="smstrack_input"></span>\n' +
    '            <span class="ui-widget"><input id="trks_input" type="text" name="trks" class="smstrack_input"></span>\n' +
    '            <span class="ui-widget"><input type="text" name="gene" id="geneorpos_input" class="gene_input" onfocus="this.value=\'\'"></span>\n' +
    '            <span class="ui-widget"><input type="text" name="tiles" id="tiles_input" class="smstiles_input" onfocus="this.value=\'\'"></span>\n' +
    '            <input type="submit" id="view_button" class="gene_input" value="view">\n' +
    '        </form>');
$('#coord_form input[name=sms]').val('all sm(s)');
$('#coord_form input[name=trks]').val('all trk(s)');
$('#coord_form input[name=gene]').val('gene or chrom:start-end');
$('#coord_form input[name=tiles]').val('tiles:400');
$('#coord_form').submit(function(e){ render(e); });

//may want to update these two to have CSV autocomplete and no auto-clear...
$.ajax({
    url: rest_host+'/sample_map',
    data: {},
    success: function(result) {
        $('#main_container').append('<div id="track_container"></div>');
        $('#track_container').append('<div id="nav_container"></div>');

        var sms = [];
        for(x in result) {
            for(s in result[x]) {
                sms.push(s);
            }
        }
        sms.sort();
        var s = 'Loaded the folowing samples:\n';
        for(var i in sms) {
            $('#track_container').append('<div id="div_'+sms[i]+'"></div>');
            s += sms[i]+'\n';
        }
        alert(s)
        scroll_sm = sms[0];
        samples = result;
        sample_names = [];
        track_names = [];
        var tracks = {};
        for(x in samples){
            sample_names.push(Object.keys(samples[x])[0]);
            for(sm in samples[x]){
                for(rg in samples[x][sm]){
                    for(seq in samples[x][sm][rg]){
                        for(t in samples[x][sm][rg][seq]){
                            tracks[t] = t;
                        }
                    }
                }
            }
        }
        sample_names = sample_names.concat(["all sm(s)"]).sort();
        for(t in tracks){
            track_names.push(tracks[t]);
        }
        track_names = track_names.concat(["all trk(s)"]).sort();

        function split( val ) {
            return val.split( /,\s*/ );
        }
        function extractLast( term ) {
            return split( term ).pop();
        }

        $( "#sms_input" )
            .on( "keydown", function( event ) {
                if ( event.keyCode === $.ui.keyCode.TAB &&
                    $( this ).autocomplete( "instance" ).menu.active ) {
                    event.preventDefault();
                }
            })
            .autocomplete({
                delay: 10,
                minLength: 1,
                source: function( request, response ) {
                    response( $.ui.autocomplete.filter(
                        sample_names, extractLast( request.term ) ) );
                },
                focus: function() {
                    return false;
                },
                select: function( event, ui ) {
                    var terms = split( this.value );
                    terms.pop();
                    terms.push( ui.item.value );
                    terms.push( "" );
                    var all = false;
                    for(var i in terms){
                        if(terms[i]=='all sm(s)'){
                            this.value = 'all sm(s)';
                            return false;
                        }
                    }
                    this.value = terms.join(",");
                    return false;
                }
            });

        $( "#trks_input" )
            .on( "keydown", function( event ) {
                if ( event.keyCode === $.ui.keyCode.TAB &&
                    $( this ).autocomplete( "instance" ).menu.active ) {
                    event.preventDefault();
                }
            })
            .autocomplete({
                delay: 10,
                minLength: 0,
                source: function( request, response ) {
                    response( $.ui.autocomplete.filter(
                        track_names, extractLast( request.term ) ) );
                },
                focus: function() {
                    return false;
                },
                select: function( event, ui ) {
                    var terms = split( this.value );
                    terms.pop();
                    terms.push( ui.item.value );
                    terms.push( "" );
                    var all = false;
                    for(var i in terms){
                        if(terms[i]=='all trk(s)'){
                            this.value = 'all trk(s)';
                            return false;
                        }
                    }
                    this.value = terms.join(",");
                    return false;
                }
            });
    }
});

$.ajax({
    url: rest_host+'/gene_map',
    data: {},
    success: function (result) {
        genes = Object.keys(result).sort();
        //console.log(genes);
        $( function() {
            $( "#geneorpos_input" ).autocomplete({
                source: genes,
                minLength:3,
                delay: 10
            });
        });
    }
});

$.ajax({
    url: rest_host+'/ref_map',
    data: {},
    success: function (result) {
        ref_seqs = result;
    }
});