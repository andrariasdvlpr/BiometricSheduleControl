function rejectsubmit() {
    submitAction = document.getElementById("manageForm");
    submitAction.action ="reject";
    submitAction.submit();
}
function restControls(option) {
    var imagCheck = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAB30lEQVRIx9WVP4sTURTFz3mYBbWzyRvURgkWiwgu9n6CnbXYJkTEJp9AUXFhR1A2soK9dVyEFG78BPkAayWpgpXBeSksV/wT51hsNvucmWQmwWZv9Zj75nfO+3cvcNqD85JNoXLobEhqg+KagMsARGAo6qPE/fPWdd8QvxcWqLtqaBKzC6o216I4SEzycM+OuqUEIsF8joMXoh4vsBOS2KoF8VZEJH7CpGcuAQcAknoyiIPnc1dQd9XQiO9RcDY58QnAdQACtdG2ow+ZFTSFiknM7hLwztC6mwAiAIT4erO/upIROHQ2LDzQbLwbWlfvEWPP2JWVC9/WMwJGvLOE87s9YtyIbQRge7rvVJgRELXm/y2gX8Z5Gj6B3coIALjojaNf1t0A9XYR53ksX0De+GyH+POzOrqXEunMdX4SSZ7AV2/8qBHblifSXgAOeqwzJ191APFaSuRHmy66rdF9ACgDPzoCHWRWIHE/Z+52I7ZRjxiXhU9Y07o0fVRNofI9Dvoz3sKzY8EiOMTBuSBePa6w/6tUTPEzSwUA7NlRl+LLJeGQ2PLhGQEAuBrETyXu4N9rW8wWd2pBvJVOzNyKhquuIzGvyjQcmORB2nmhAJDbMi9NUl9w1DK7RS3z9MdfQWjuJPV6wyoAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjAtMDYtMDdUMjA6NDk6MjkrMDA6MDAWpkvfAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIwLTA2LTA3VDIwOjQ5OjI5KzAwOjAwZ/vzYwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAASUVORK5CYII=";
    var imagError = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAA7EAAAOxAGVKw4bAAABkklEQVRIx8XVPWsUURTG8d+sGrJNxEojbCEWQnZtfAkJ2BmwiigKVn4FxQ8gmF4hiFjYWvmCGIKFhZ8gpJzBFGKRwpdGkmajMY5FZjaZyzqzWSfk6ebce57/uedyz3DQiplMGA/jCeMxk1X5UdniJ45usIJjeI3H2dId3MSPUc6cZm0oQMw87pbtSZnvcG9PgM+Mdrmc8haHK7rwO+Jakw+n2CgFxDzAFZzDSFV/A/3CMt63mcuDjWDTVUwNYS7Lmc48egoB38oc2kTtinsLPULA1yEqD1Xw6AFWaeJsDYBO5rUDSGms8xznawBcXOdFyqEeIOYRbtRgnms25mGhRfulKG9RwssaT7E4wfWIrUZG+TPGbSzVYL40xq2IrUKLWnQjkrLMAd9B3KKbfxTuIOVEDScoeISD7HhpaaQDAAoeIWABP20PuyN7rLw37HYH+/ZzleYaMxFv/Oe47vsOWnQ7LOJpVdkpTyZ418/8n4BcI9zHF2ziVcqlBhfwzLbh981ds38oJUx95GSf+EA//X3XX1EOWQXEcMZTAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDIwLTA2LTExVDIzOjI2OjEzKzAwOjAwTBpM1QAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyMC0wNi0xMVQyMzoyNjoxMyswMDowMD1H9GkAAAAZdEVYdFNvZnR3YXJlAHd3dy5pbmtzY2FwZS5vcmeb7jwaAAAAAElFTkSuQmCC";
    imag= imagCheck;
    if(option.localeCompare("rejected")==0){imag = imagError;}
    var table = document.getElementById("table-records");
    table.innerHTML="";
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){
                var user = JSON.parse(this.responseText); 
                var x; 
                for(x in user){
                    var img = document.createElement("IMG");
                    img.setAttribute('src', 'data:image/png;base64,'+imag);
                    var row = table.insertRow(x);
                    var cell1 = row.insertCell(0);
                    var cell2 = row.insertCell(1);
                    var cell3 = row.insertCell(2);
                    var cell4 = row.insertCell(3);
                    var string = "getControlData("+user[x].sc_id+")";
                    row.setAttribute("onClick", string);
                    cell1.innerHTML = user[x].employee_id;
                    cell2.innerHTML = user[x].sc_date.replace("T","  ").replace("Z","");
                    cell3.innerHTML = user[x].sc_distance.toFixed(5);
                    cell4.appendChild(img);
                }

            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.16/bsc/rest-control-user/"+document.getElementById("selectFilter").value+"/"+option;
    xmlhttp.open("GET", url, true);
    xmlhttp.setRequestHeader("Authorization", "Token "+ getCookie("FC_SESSIONID") )
    xmlhttp.send();
}

function getControlData(recordid){
    var xmlhttp = new XMLHttpRequest();
    
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){

                var user = JSON.parse(this.responseText);  
                document.getElementById("record_id").value = recordid;
                document.getElementById("employee_id").value = user.employee_id;
                document.getElementById("firstname").value = user.first_name;
                document.getElementById("lastname").value = user.last_name;
                document.getElementById("score").value = user.score;
                document.getElementById("type").value = user.type;
                document.getElementById("dateControl").value = user.date_control.replace("T","  ").replace("Z","");
                if(user.template.localeCompare("null")!=0){
                    document.getElementById('imgPopUp').setAttribute('src', 'data:image/png;base64,'+user.template);
                }
                else{
                    document.getElementById('imgPopUp').setAttribute('src', '/static/styleResources/account-circle.png');
                }
                if(user.image.localeCompare("null")!=0){
                    document.getElementById('imgPopUp2').setAttribute('src', 'data:image/png;base64,'+user.image);
                }
                else{
                    document.getElementById('imgPopUp2').setAttribute('src', '/static/styleResources/account-circle.png');
                }
                document.getElementById("address").value = user.location.address;
                document.getElementById("latitude").value = user.location.latitude;
                document.getElementById("longitude").value = user.location.longitude;
                document.getElementById('map-container').innerHTML ="<div class ='img-template-m' id='map' style='border-radius: 0;'></div>";
                document.getElementById('modal-wrapper').style.display='flex';
                
                let mymap = L.map('map').setView([user.location.latitude, user.location.longitude], 13);
                L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors',
                    maxZoom: 18
                }).addTo(mymap);
                L.control.scale().addTo(mymap);
                L.marker([user.location.latitude, user.location.longitude]).addTo(mymap);

            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.16/bsc/rest-control/"+recordid;
    xmlhttp.open("GET", url, true);
    xmlhttp.setRequestHeader("Authorization", "Token "+ getCookie("FC_SESSIONID") )
    xmlhttp.send();
}

// Initialize and add the map
function proving() {
    var map = document.getElementById("map");
    map.style.width = "500px";
    map.style.height = "500px";
    map.invalidateSize();

}

function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
      var c = ca[i];
      while (c.charAt(0) == ' ') {
        c = c.substring(1);
      }
      if (c.indexOf(name) == 0) {
        return c.substring(name.length, c.length);
      }
    }
    return "";
  }