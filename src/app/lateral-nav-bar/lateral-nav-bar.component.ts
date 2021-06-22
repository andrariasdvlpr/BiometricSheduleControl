import { Component, OnInit,Input, OnChanges, SimpleChanges, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-lateral-nav-bar',
  templateUrl: './lateral-nav-bar.component.html',
  styleUrls: ['./lateral-nav-bar.component.css']
})
export class LateralNavBarComponent implements OnInit,OnChanges {

  @Input() show: boolean = false;
  @Output() closeEvent = new EventEmitter<boolean>();


  constructor() { }

  ngOnInit(): void {
  }

  ShowDiv(hshow :boolean): void {
    this.show = hshow;
    this.closeEvent.emit(hshow);
  }

  ngOnChanges(changes: SimpleChanges) {
    console.log(changes.show)
  }

}
