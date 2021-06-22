import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HistoryFragmentComponent } from './history-fragment.component';

describe('HistoryFragmentComponent', () => {
  let component: HistoryFragmentComponent;
  let fixture: ComponentFixture<HistoryFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HistoryFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(HistoryFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
