import { Component, OnInit, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-top-bar',
  templateUrl: './top-bar.component.html',
  styleUrls: ['./top-bar.component.css']
})
export class TopBarComponent implements OnInit {

  @Output() openEvent = new EventEmitter<boolean>();

  constructor() { }

  ngOnInit(): void {
  }

  openDrawler(value: boolean) {
    this.openEvent.emit(value);
  }
}
