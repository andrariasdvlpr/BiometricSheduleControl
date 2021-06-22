import { Component, OnInit, AfterViewInit, Inject } from '@angular/core';
import {MatDialog, MAT_DIALOG_DATA} from '@angular/material/dialog';
import * as L from 'leaflet';

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css']
})
export class MapComponent implements OnInit, AfterViewInit  {

  private map: any;
  longitude : any;
  latitude : any;
  constructor(@Inject(MAT_DIALOG_DATA) public data :{longitude: any , latitude : any}) {
    this.longitude= data.longitude;
    this.latitude = data.latitude;
   }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
    this.initMap();
  }

  private initMap(): void {
    this.map = L.map('map', {
      center: [ this.latitude, this.longitude ],
      zoom: 13
    });

    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: 'Map data &copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    });
    const marker = L.marker([this.latitude, this.longitude]);

    tiles.addTo(this.map);
    marker.addTo(this.map);
  }

}
